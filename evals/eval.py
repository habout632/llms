import tiktoken
from deepeval.benchmarks.schema import MultipleChoiceSchema
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import torch
from models.gpt2.model import GPT, GPTGenerator
from deepeval.benchmarks import MMLU, GSM8K
from deepeval.benchmarks.tasks import MMLUTask


class GPT2(DeepEvalBaseLLM):
    def __init__(
            self,
            model,
            tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema) -> str:
        try:
            prompt = prompt.strip()

            if not prompt:
                prompt = "Hello, world!"
                print("No prompt provided, using default prompt:", prompt)
                return prompt

            input_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            output_ids = self.model(input_ids)

            # output_ids = torch.argmax(output_ids, dim=-1)

            output_ids = torch.argmax(output_ids, dim=-1)
            output_ids = output_ids.cpu().numpy()

            output_text = self.tokenizer.decode(output_ids[0])
            return output_text
        except Exception as e:
            print(f"Error occurred during generation: {e}")
            return ''


    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    # def batch_generate(self, promtps: List[str]) -> List[str]:
    #     model = self.load_model()
    #     device = "cuda"  # the device to load the model onto
    #
    #     model_inputs = self.tokenizer(promtps, return_tensors="pt").to(device)
    #     model.to(device)
    #
    #     generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    #     return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return "GPT2"


# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

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

state_dict = torch.load('../models/gpt2/best.pt')
model.load_state_dict(state_dict)
# model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

gpt2 = GPT2(model=model, tokenizer=tokenizer)

# config = {
#     "vocab_size": 50257,
#     "emb_dim": 768,
#     "context_length": 1024,
#     "drop_rate": 0.1,
#     "n_heads": 12,
#     "qkv_bias": True
# }
# model = GPT(config)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# generator = GPTGenerator(model, tokenizer, config)


print(gpt2.generate("Write me a joke", MultipleChoiceSchema))

# Define benchmark with specific tasks and shots
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY]
    # n_shots=3
)

# benchmark = GSM8K(
#     n_shots=3
# )

from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask

# # Define benchmark with specific tasks and shots
# benchmark = HellaSwag(
#     tasks=[HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING],
#     n_shots=5
# )


# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=gpt2)
print(benchmark.overall_score)
