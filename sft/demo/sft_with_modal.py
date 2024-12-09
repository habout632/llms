import os
import sys
import subprocess
import torch
import modal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TrainingConfig:
    # Modal 配置
    APP_NAME: str = "llama_factory"
    VOLUME_NAME: str = "sft"
    # GPU_CONFIG: Any = modal.gpu.A100(size="40GB", count=1)
    GPU_CONFIG: Any = modal.gpu.A10G()
    TIMEOUT_SECONDS: int = 30 * 60

    # 文件名配置
    DATASET_FILENAME: str = "alpaca_zh_demo.json"
    CONFIG_FILENAME: str = "sft.yaml"
    DATASET_INFO_FILENAME: str = "dataset_info.json"

    # 目录配置
    ROOT_DIR: str = "/root"
    LLAMA_FACTORY_DIR: str = "/root/LLaMA-Factory"
    DATA_DIR: str = "/root/LLaMA-Factory/data"

    @property
    def DATASET_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, self.DATASET_FILENAME)

    @property
    def CONFIG_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, self.CONFIG_FILENAME)

    @property
    def DATASET_INFO_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, self.DATASET_INFO_FILENAME)

    # URL 配置
    DATASET_URL: str = "https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/refs/heads/main/data/alpaca_zh_demo.json"
    DATASET_INFO_URL: str = "https://raw.githubusercontent.com/habout632/llms/refs/heads/main/sft/demo/dataset_info.json"
    CONFIG_URL: str = "https://raw.githubusercontent.com/habout632/llms/refs/heads/main/sft/demo/sft.yaml"

    # Python 包配置
    PYTHON_PACKAGES: List[str] = field(
        default_factory=lambda: [
            "numpy<3",
            "tqdm",
            "huggingface_hub",
            "transformers",
            "metrics"
        ]
    )

    # 系统包配置
    SYSTEM_PACKAGES: List[str] = field(
        default_factory=lambda: ["wget", "git"]
    )

    # PyTorch 配置
    TORCH_INDEX_URL: str = "https://download.pytorch.org/whl/nightly/cu124"

    def get_setup_commands(self) -> List[str]:
        """获取环境设置命令"""
        return [
            f"cd {self.ROOT_DIR} && git clone https://github.com/hiyouga/LLaMA-Factory.git",
            f"cd {self.LLAMA_FACTORY_DIR} && pip install -e ."
        ]

    def get_download_commands(self) -> List[str]:
        """获取文件下载命令"""
        return [
            f"mkdir -p {self.DATA_DIR}",
            f"wget -O {self.DATASET_INFO_PATH} {self.DATASET_INFO_URL}",
            f"wget -O {self.DATASET_PATH} {self.DATASET_URL}",
            f"wget -O {self.CONFIG_PATH} {self.CONFIG_URL}"
        ]



config = TrainingConfig()

app = modal.App(config.APP_NAME)
vol = modal.Volume.from_name(config.VOLUME_NAME)


def create_image() -> modal.Image:
    """创建 Modal image"""
    return (
        modal.Image.debian_slim(python_version="3.12.7")
        .pip_install(config.PYTHON_PACKAGES)
        .pip_install(
            "torch",
            pre=True,
            index_url=config.TORCH_INDEX_URL,
        )
        .apt_install(config.SYSTEM_PACKAGES)
        .run_commands(config.get_setup_commands())
        .run_commands(config.get_download_commands(), force_build=True)
    )

image = create_image()

def check_environment():
    """检查训练环境"""
    print("\n=== 环境检查 ===")
    print("当前工作目录:", os.getcwd())
    print("CUDA is available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, "GB")

def check_files():
    """检查必要文件"""
    print("\n=== 文件检查 ===")
    print("LLaMA-Factory 目录内容:")
    subprocess.run(["ls", "-la", config.LLAMA_FACTORY_DIR])
    print("\n数据目录内容:")
    subprocess.run(["ls", "-la", config.DATA_DIR])
    print("\n配置文件内容:")
    subprocess.run(["cat", f"{config.DATA_DIR}/sft.yaml"])

@app.function(
    image=image,
    gpu=config.GPU_CONFIG,
    volumes={"/data": vol},
    timeout=config.TIMEOUT_SECONDS,
)
def train():
    check_environment()
    check_files()

    print("\n=== 开始训练 ===")
    try:
        which_result = subprocess.run(
            ["which", "llamafactory-cli"],
            capture_output=True,
            text=True
        )
        print("llamafactory-cli 路径:", which_result.stdout)

        result = subprocess.run(
            ["llamafactory-cli", "train", f"{config.DATA_DIR}/sft.yaml"],
            cwd=config.LLAMA_FACTORY_DIR,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        print("\n命令输出:")
        print(result.stdout)
        print("\n错误输出:")
        print(result.stderr)

        return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"训练失败，返回码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        raise e
    except Exception as e:
        print(f"发生未预期的错误: {str(e)}")
        raise e

@app.local_entrypoint()
def main():

    print("开始在远程服务器上训练")
    try:
        returncode = train.remote()
        print(f"训练{'成功' if returncode == 0 else '失败'}")
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise e