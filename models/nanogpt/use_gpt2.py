import torch
from transformers import GPT2Tokenizer
from models.gpt2.model import GPT, GPTGenerator
from safetensors.torch import load_file

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 1024,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
model = GPT(GPT_CONFIG_124M)


# 2. Load the weights from .pt file
# state_dict = torch.load('../gpt2/best.pt')
# model.load_state_dict(state_dict)

# Load safetensors file directly
state_dict = load_file("../gpt2/gpt2/model.safetensors")

# Create and load your model
model.load_state_dict(state_dict)

# 3. Set model parameters
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()  # Set to evaluation mode
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# GPT2 doesn't have a chat template, so we'll format the text directly
text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
model_inputs = tokenizer(text)["input_ids"]

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
#
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
generator = GPTGenerator(model, tokenizer, GPT_CONFIG_124M)
print(generator.generate("Hello, world!"))