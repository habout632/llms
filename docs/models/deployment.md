
# Hardware Requirements
训练的时候 用 8 * H100 80GB
部署的时候 用A100 80GB
线上部署的时候 不用量化版本 使用bf16

## Qwen2.5
7B 15.6GB
* 14B 29.7GB 1 * A100  
32B 66.3GB 1 * A100  
72B 145.78GB

## Qwen2 VL
2B - 
7B  16.48 GB(16.07 - 44.08) 1 * A100
* 72B  4 * 38 = 152GB(138.74 - 204.33GB) 4 * A100 

## Deepseek

700+ GB 显存


# vLLM

# LLamaFactory

#  deepspeed 