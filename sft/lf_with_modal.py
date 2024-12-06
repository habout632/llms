import os

import modal

app = modal.App("med")
vol = modal.Volume.from_name("sft")
# vol = modal.Volume.from_name("modded-nanogpt-logs")


# @app.function(volumes={"/data": vol})
TARGET = "/root/"
# MODEL_URL = f"https://raw.githubusercontent.com/habout632/llms/refs/heads/main/models/nanogpt/nanogpt.py"
DATASET_URL = f"https://huggingface.co/datasets/habout632/medicine_test/resolve/main/medbench_20241204.jsonl"
CONFIG_URL = f"https://raw.githubusercontent.com/habout632/llms/refs/heads/main/sft/sft.yaml"

image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("numpy<3", "tqdm", "huggingface_hub", "transformers", "metrics")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
    .apt_install("wget", "git")
    # .run_commands([f"git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git"],force_build=True)
    # Install LLaMA Factory - modified installation steps
    # 安装 LLaMA Factory - 修改了安装步骤
    .run_commands([
        "cd /root && git clone https://github.com/hiyouga/LLaMA-Factory.git",
        "cd /root/LLaMA-Factory && pip install -e ."  # 确保在正确的目录下执行安装
    ])
    # Download required files
    .run_commands([
        f"wget -O {TARGET}medbench_20241204.jsonl {DATASET_URL}",
        f"wget -O {TARGET}sft.yaml {CONFIG_URL}"
    ], force_build=True)
    .run_commands([
        f"llamafactory-cli train /root/sft.yaml"
    ])
    # .run_commands([f"wget -O {TARGET + 'medbench_20241204.jsonl'} {DATASET_URL}"])
    # .run_commands([f"wget -O {TARGET + 'sft.yaml'} {CONFIG_URL}"], force_build=True)
    # .run_commands([f"wget -O {TARGET + 'medbench_20241204.jsonl'} {DATASET_URL}"], force_build=True)
    # .env({"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"})
    # .env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})
    # .env({"HF_HOME": "1"})
)


@app.function(
    image=image,
    # gpu=f"H100:{N_H100}",
    # gpu=f"A10G",
    gpu=modal.gpu.A100(size="40GB", count=1),
    # volumes={
    #     TARGET + "data": data,
    #     TARGET + "logs": logs,
    #     # mount the caches of torch.compile and friends
    #     "/root/.nv": modal.Volume.from_name("nanogpt-nv-cache", create_if_missing=True),
    #     "/root/.triton": modal.Volume.from_name(
    #         "nanogpt-triton-cache", create_if_missing=True
    #     ),
    #     "/root/.inductor-cache": modal.Volume.from_name(
    #         "nanogpt-inductor-cache", create_if_missing=True
    #     ),
    # },
    volumes={"/data": vol},
    timeout=30 * 60,
)
def run():
    # os.environ['HF_HOME'] = '/data/models'

    import subprocess

    # First verify the directory exists
    print("Current directory contents:")
    dirs = subprocess.run(["ls", "-la", "/root"], capture_output=True, text=True)
    print(dirs.stdout)
    #
    # # Run command and wait for it to complete
    # subprocess.run(["ls", "-l"])

    # Capture output
    # llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
    # result = subprocess.run(["cd", "/root/LLaMA-Factory"], capture_output=True, text=True)
    result = subprocess.run(["llamafactory-cli", "train", "/root/sft.yaml"],
                            capture_output=True, text=True)
    print(result.stdout)

    # Shell commands
    subprocess.run("echo $HOME", shell=True)


@app.local_entrypoint()
def main():
    print("get data on remote server")
    run.remote()
