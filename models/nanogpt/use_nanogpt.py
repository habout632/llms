
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
    .run_commands([f"wget -O {TARGET + 'nanogpt.py'} {SCRIPT_URL}"])
    # .run_commands([f"wget -O {TARGET + 'nanogpt.py'} {SCRIPT_URL}"], force_build=True)
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
    import torch.nn.functional as F
    # from models.gpt2.model import GPT, GPTGenerator
    # from safetensors.torch import load_file
    #

    from nanogpt import GPT, GPTConfig
    num_vocab = 50304
    max_new_tokens = 11
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))

    # 2. Load the weights from .pt file
    state_dict = torch.load("/data/80a44e2e-4623-4a6a-ac88-1b5bb00bcbf3/state_step001875.pt")
    # state_dict = torch.load("/data/cb5594a7-074a-4daa-8d34-bb2610bbdece/state_step000010.pt")

    # Remove '_orig_mod.' prefix from keys
    new_state_dict = {}
    for key, value in state_dict['model'].items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict['model'], strict=False)

    print(f"model state dict:{state_dict.keys()}")

    # 3. Set model parameters
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()  # Set to evaluation mode
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    # prompt = "Give me a short introduction to large language model."
    # messages = [
    #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]
    #
    # # GPT2 doesn't have a chat template, so we'll format the text directly
    # text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
    # def generate_text(model, text, max_new_tokens=10):
    #     # Tokenize input text
    #     input_ids = tokenizer(text)["input_ids"]
    #     idx = torch.tensor(input_ids, dtype=torch.long, device='cuda')  # [seq_len]
    #
    #     # Generate tokens
    #     for _ in range(max_new_tokens):
    #         # Crop if needed
    #         if len(idx) > 1024:
    #             idx = idx[-1024:]
    #
    #         # Forward pass and get next token
    #         with torch.no_grad():
    #             logits = model(idx)  # Model expects [seq_len]
    #             next_token = torch.argmax(logits[0, -1, :])  # Get last token prediction
    #
    #             # Append new token (both tensors are 1D)
    #             idx = torch.cat((idx, next_token.unsqueeze(0)), dim=0)  # [seq_len + 1]
    #
    #         # Optional: Stop at EOS token
    #         if next_token.item() == tokenizer.eos_token_id:
    #             break
    #
    #     # Decode sequence
    #     output_text = tokenizer.decode(idx, skip_special_tokens=True)
    #     return output_text

    def generate_text(model, text, max_new_tokens=10, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
        input_ids = tokenizer(text)["input_ids"]
        idx = torch.tensor(input_ids, dtype=torch.long, device='cuda')

        # Keep track of generated tokens for repetition penalty
        generated = idx.tolist()

        def apply_sampling(logits):
            # Apply temperature
            logits = logits / temperature

            # Apply repetition penalty
            if len(generated) > 0:
                for old_token in set(generated):
                    logits[old_token] /= repetition_penalty

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            return next_token

        for _ in range(max_new_tokens):
            if len(idx) > 1024:
                idx = idx[-1024:]

            with torch.no_grad():
                logits = model(idx)
                next_token_logits = logits[0, -1, :]
                next_token = apply_sampling(next_token_logits)

                # Add to generated tokens list for repetition penalty
                generated.append(next_token.item())

                # Concatenate new token
                idx = torch.cat((idx, next_token), dim=0)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(idx, skip_special_tokens=True)

    # # Usage:
    # text = "Once upon a time"
    # output = generate_text(model, text, max_new_tokens=200)
    # print(output)
    # Usage example:
    text = "tell me a story"
    output = generate_text(
        model,
        text,
        max_new_tokens=50,
        temperature=0.7,  # Higher = more random, Lower = more focused
        top_k=50,  # Keep only top k tokens
        top_p=0.9,  # Nucleus sampling threshold
        repetition_penalty=1.2  # Penalty for repeating tokens
    )
    print(output)

    #
    # text = "Once upon a time"
    #
    # # # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # input_ids = tokenizer(text)["input_ids"]
    # print(input_ids)
    # #
    # # input_ids = torch.tensor(input_ids).cuda()
    # #
    # # output_ids = model(input_ids)
    # #
    # # # output_ids = torch.argmax(output_ids, dim=-1)
    # #
    # # output_ids = torch.argmax(output_ids, dim=-1)
    # # output_ids = output_ids.cpu().numpy()
    # # print(f"output ids:{output_ids}")
    # #
    # # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # # print(output_text)
    #
    # #     f.write("hello")
    # # vol.commit()  # Needed to make sure all changes are persisted
    #
    # # size = os.path.getsize("/data/cb5594a7-074a-4daa-8d34-bb2610bbdece/state_step000010.pt")
    # # print(f"File size: {size} bytes")
    #
    # # Generate max_new_tokens tokens
    # idx = torch.tensor(input_ids).cuda()
    # for _ in range(max_new_tokens):
    #     # If sequence length exceeds model's context size, crop it
    #     if len(idx) > 1024:
    #         idx = idx[-1024:]
    #
    #     # Forward pass
    #     with torch.no_grad():
    #         logits = model(idx)
    #
    #     # Get logits of the last token only
    #     # logits = logits[-1, :]
    #     output_ids = torch.argmax(logits, dim=-1)
    #     # output_ids = output_ids.cpu().numpy()
    #     idx_next = output_ids.squeeze()
    #     # idx_next = idx_next[-1]
    #
    #     # # Optional: Apply temperature and top-k sampling
    #     # probs = F.softmax(logits / 0.7, dim=-1)  # temperature = 0.7
    #     # idx_next = torch.multinomial(probs, num_samples=1)
    #
    #     # Append the new token
    #     print(f"idx shape:{idx.shape}")
    #     print(f"idx next shape:{idx_next.shape}")
    #     idx = torch.cat((idx, idx_next), dim=-1)
    #
    #     # Optional: Stop if we generate an EOS token
    #     # if idx_next == tokenizer.eos_token_id:
    #     #     break
    #
    # # Decode the generated tokens
    # output_text = tokenizer.decode(idx, skip_special_tokens=True)
    # print(output_text)

@app.local_entrypoint()
def main():
    print("get data on remote server")
    run.remote()
