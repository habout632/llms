# 0. imports
import torch
from transformers import GPT2Tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
#
# from huggingface_hub import configure_http_backend
# configure_http_backend(
#     http="https://hf-mirror.com",
#     https="https://hf-mirror.com"
# )

model_name = "Qwen/Qwen2.5-0.5B-instruct"  # 或其他Qwen2.5模型
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
auto_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)


# 1. load a pretrained model
print(f"loading pretrained model")
model = AutoModelForCausalLMWithValueHead.from_pretrained(auto_model)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
print("initialize trainer")
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 3. encode a query
print("encode query")
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
print("generate response")
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
print("train model")
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
print(train_stats)