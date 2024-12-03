import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import modal
from huggingface_hub import PyTorchModelHubMixin
from torch.nn.parallel import DistributedDataParallel as DDP

# from models.nanogpt.try_gpt2 import GPTConfig, GPT
from models.gpt2.model import GPT


app = modal.App("try-ddp")
image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("numpy<3", "huggingface_hub", "transformers")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# On Windows platform, the torch.distributed pacage only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group gloo for cpu , nccl for gpu
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    # model = ToyModel().to(rank)

    # num_vocab = 50304
    # model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768)).to(rank)
    #
    config = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "drop_rate": 0.1,
        "n_heads": 12,
        "qkv_bias": True
    }
    model = GPT(config).to(rank)
    # model = model.cuda()

    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs = ddp_model(torch.randn(20, 10))
    outputs = ddp_model(torch.randn(20, 768))
    # labels = torch.randn(20, 5).to(rank)
    labels = torch.randn(20, 768).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    ddp_model.module.save_pretrained("gpt2")

    # push to the hub
    ddp_model.module.push_to_hub(
        repo_id="habout632/gpt2",
        token="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")

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


# @app.function(
#     gpu=modal.gpu.A10G(count=3),
#     image=image
# )
def run_demo():
    world_size = torch.cuda.device_count()
    mp.spawn(demo_basic,
             args=(world_size,),
             nprocs=world_size,
             join=True)


# @app.local_entrypoint()
# def main():
#     run_demo.remote()

if __name__ == '__main__':
    run_demo()
