from transformers import AutoModelForCausalLM, GPT2Tokenizer

model_name = "habout632/gpt2"

model = AutoModelForCausalLM.from_pretrained(
    "../gpt2/gpt2",
    torch_dtype="auto",
    device_map="auto"
)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# GPT2 doesn't have a chat template, so we'll format the text directly
text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]