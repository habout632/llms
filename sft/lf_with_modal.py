import os

import modal

app = modal.App("med")
vol = modal.Volume.from_name("sft")
# vol = modal.Volume.from_name("modded-nanogpt-logs")


# @app.function(volumes={"/data": vol})
TARGET = "/root/"
# MODEL_URL = f"https://raw.githubusercontent.com/habout632/llms/refs/heads/main/models/nanogpt/nanogpt.py"
DATASET_URL = f"https://huggingface.co/datasets/habout632/medicine_test/resolve/main/medbench_20241204.jsonl"
DATASET_INFO_URL = f"https://raw.githubusercontent.com/habout632/llms/refs/heads/main/sft/dataset_info.json"
CONFIG_URL = f"https://raw.githubusercontent.com/habout632/llms/refs/heads/main/sft/sft.yaml"

# Fixed JSON string with proper escaping
# DATASET_INFO = '''
# {
#     "medbench": {
#         "file_name": "medbench_20241204.jsonl",
#         "file_sha1": null,
#         "columns": {
#             "prompt": "instruction",
#             "query": "input",
#             "response": "output"
#         }
#     }
# }
# '''.strip()



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
    # .run_commands([
    #     f"wget -O {TARGET}medbench_20241204.jsonl {DATASET_URL}",
    #     f"wget -O {TARGET}sft.yaml {CONFIG_URL}"
    # ], force_build=True)
    .run_commands([
        "mkdir -p /root/LLaMA-Factory/data",
        # 下载数据集和配置文件
        f"wget -O /root/LLaMA-Factory/data/dataset_info.json {DATASET_INFO_URL}",
        f"wget -O /root/LLaMA-Factory/data/medbench_20241204.jsonl {DATASET_URL}",
        f"wget -O /root/LLaMA-Factory/data/sft.yaml {CONFIG_URL}"
    ], force_build=True)
    # .run_commands([
    #     f"llamafactory-cli train /root/LLaMA-Factory/data/sft.yaml"
    # ])
    # .run_commands([f"wget -O {TARGET + 'medbench_20241204.jsonl'} {DATASET_URL}"])
    # .run_commands([f"wget -O {TARGET + 'sft.yaml'} {CONFIG_URL}"], force_build=True)
    # .run_commands([f"wget -O {TARGET + 'medbench_20241204.jsonl'} {DATASET_URL}"], force_build=True)
    # .env({"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"})
    # .env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})
    # .env({"HF_HOME": "1"})
)


@app.function(
    image=image,
    # gpu=f"H100:1",
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
    import torch
    import sys
    import subprocess

    # 验证 GPU
    # 验证环境
    print("\n=== 环境检查 ===")
    print("当前工作目录:", os.getcwd())
    print("CUDA is available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

    print("\n=== 文件检查 ===")
    print("LLaMA-Factory 目录内容:")
    subprocess.run(["ls", "-la", "/root/LLaMA-Factory"])

    print("\n数据目录内容:")
    subprocess.run(["ls", "-la", "/root/LLaMA-Factory/data"])

    print("\n配置文件内容:")
    subprocess.run(["cat", "/root/LLaMA-Factory/data/sft.yaml"])



    # # First verify the directory exists
    # print("Current directory contents:")
    # dirs = subprocess.run(["ls", "-la", "/root"], capture_output=True, text=True)
    # print(dirs.stdout)
    #
    # # Run command and wait for it to complete
    # subprocess.run(["ls", "-l"])

    # Capture output
    # llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
    # result = subprocess.run(["cd", "/root/LLaMA-Factory"], capture_output=True, text=True)

    print("Starting training...")

    # 使用 subprocess.run 但不捕获输出，而是直接打印到控制台
    try:
        # 首先验证命令是否存在
        which_result = subprocess.run(["which", "llamafactory-cli"],
                                    capture_output=True,
                                    text=True)
        print("llamafactory-cli 路径:", which_result.stdout)

        result = subprocess.run(
            ["llamafactory-cli", "train", "/root/LLaMA-Factory/data/sft.yaml"],
            cwd="/root/LLaMA-Factory",
            check=True,  # 这会在命令失败时抛出异常
            text=True,
            stdout=sys.stdout,  # 直接输出到控制台
            stderr=sys.stderr
        )
        print("\n命令输出:")
        print(result.stdout)
        print("\n错误输出:")
        print(result.stderr)

        if result.returncode != 0:
            print(f"\n命令失败，返回码: {result.returncode}")
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                result.stdout,
                result.stderr
            )
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise e
    # Shell commands
    subprocess.run("echo $HOME", shell=True)


@app.local_entrypoint()
def main():
    print("get data on remote server")
    run.remote()
