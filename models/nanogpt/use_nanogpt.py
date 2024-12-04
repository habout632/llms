
#
# num_vocab = 50304
# model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
#
#
# # 2. Load the weights from .pt file
# state_dict = torch.load('./state_step000010.pt')
# model.load_state_dict(state_dict)
#
# # Load safetensors file directly
# # state_dict = load_file("../gpt2/gpt2/model.safetensors")
#
# # Create and load your model
# # model.load_state_dict(state_dict)
#
# # 3. Set model parameters
# model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# model.eval()  # Set to evaluation mode
# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
#
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
#
# # GPT2 doesn't have a chat template, so we'll format the text directly
# text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
#
# # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# model_inputs = tokenizer(text)["input_ids"]
#
# # generated_ids = model.generate(
# #     **model_inputs,
# #     max_new_tokens=512
# # )
# # generated_ids = [
# #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# # ]
# #
# # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# generator = GPTGenerator(model, tokenizer, GPT_CONFIG_124M)
# print(generator.generate("Hello, world!"))
import os

import modal

app = modal.App("use-nanogpt")
vol = modal.Volume.from_name("modded-nanogpt-logs")


# @app.function(volumes={"/data": vol})
TARGET = "/root/"
SCRIPT_URL = f"https://raw.githubusercontent.com/habout632/llms/refs/heads/main/models/nanogpt/nanogpt.py"

image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("numpy<3", "tqdm", "huggingface_hub", "transformers")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
    .apt_install("wget")
    .run_commands([f"wget -O {TARGET + 'nanogpt.py'} {SCRIPT_URL}"], force_build=True)
    .env({"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"})
    .env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})
)

@app.function(
    image=image,
    # gpu=f"H100:{N_H100}",
    gpu=f"A10G",
    # gpu=modal.gpu.A100(size="40GB", count=N_H100),
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
    import torch
    from transformers import GPT2Tokenizer
    # from models.gpt2.model import GPT, GPTGenerator
    # from safetensors.torch import load_file
    #

    from nanogpt import GPT, GPTConfig
    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))

    # 2. Load the weights from .pt file
    state_dict = torch.load("/data/cb5594a7-074a-4daa-8d34-bb2610bbdece/state_step000010.pt")
    model.load_state_dict(state_dict['model'], strict=False)

    print(f"model state dict:{state_dict.keys()}")

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
    input_ids = tokenizer(text)["input_ids"]
    print(input_ids)

    output_ids = model(input_ids)

    # output_ids = torch.argmax(output_ids, dim=-1)

    output_ids = torch.argmax(output_ids, dim=-1)
    output_ids = output_ids.cpu().numpy()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

    #     f.write("hello")
    # vol.commit()  # Needed to make sure all changes are persisted

    # size = os.path.getsize("/data/cb5594a7-074a-4daa-8d34-bb2610bbdece/state_step000010.pt")
    # print(f"File size: {size} bytes")


@app.local_entrypoint()
def main():
    print("get data on remote server")
    run.remote()
