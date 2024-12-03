# Train GPT-2 in five minutes -- for free
#
# ```bash
# pip install modal
# modal setup
# modal run wrapper.py
# ```
#
# Note that the end-to-end latency the first time is more like 25 minutes:
# - five minutes to install Torch (rip)
# - two minutes to download the pre-tokenized dataset
# - ten minutes to compile the model with torch.compile
# - five minutes to train the model
#
# On subsequent invocations, the first two steps are not repeated and the compile latency is cut in half.

from pathlib import Path

import modal


app = modal.App("modded-nanogpt")


REPO_ROOT = Path(__file__).parent
TARGET = "/root/"

N_H100 = 1

COMMIT_SHA = "cbc099dd73291fbd51f08b7b6f9360420f511890"
SCRIPT_URL = f"https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/{COMMIT_SHA}/train_gpt2.py"

image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("numpy<3", "tqdm", "huggingface_hub")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
    .apt_install("wget")
    .run_commands([f"wget -O {TARGET + 'train_gpt2.py'} {SCRIPT_URL}"])
    .env({"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"})
    .env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})
)

data = modal.Volume.from_name("fineweb", create_if_missing=True)
logs = modal.Volume.from_name("modded-nanogpt-logs", create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(volumes={"/data": data}, image=download_image)
def get_data(num_chunks: int = 2):
    # modified from the original in KellerJordan/modded-nanogpt
    import os
    from huggingface_hub import hf_hub_download

    # Download the GPT-2 tokens of Fineweb10B from huggingface. This
    # saves about an hour of startup time compared to regenerating them.
    def get(fname):
        local_dir = os.path.join("/data", "fineweb10B")
        if not os.path.exists(os.path.join(local_dir, fname)):
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=local_dir,
            )

    get("fineweb_val_%06d.bin" % 0)

    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)


@app.function(
    image=image,
    # gpu=f"H100:{N_H100}",
    # gpu=f"A10G:{N_H100}",
    gpu=modal.gpu.A100(size="40GB", count=N_H100),
    volumes={
        TARGET + "data": data,
        TARGET + "logs": logs,
        # mount the caches of torch.compile and friends
        "/root/.nv": modal.Volume.from_name("nanogpt-nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name(
            "nanogpt-triton-cache", create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "nanogpt-inductor-cache", create_if_missing=True
        ),
    },
    timeout=30 * 60,
)
def train():
    import os
    import subprocess

    os.chdir(TARGET)
    # makes the torch compile step less boring
    os.environ["TORCH_LOGS"] = "dynamo,graph"

    subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={N_H100}", "train_gpt2.py"]
    )


@app.local_entrypoint()
def main():
    print("get data on remote server")
    get_data.remote()
    print("start training on remote server")
    train.remote()
    # print("get data on remote server")
    # get_data.local()
    # print("start training on remote server")
    # train.local()
