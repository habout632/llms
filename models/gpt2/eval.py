# Evaluate the GPT model on the following llm_datasets:
# ARC (25-shot)
# HellaSwag (10-shot)
# MMLU (5-shot)
# TruthfulQA (0-shot)

import torch
from models.gpt2.model import GPT
from llm_datasets import load_dataset

from lm_eval import evaluator, tasks

# List available tasks
print(tasks.ALL_TASKS)

# Evaluate your model
results = evaluator.simple_evaluate(
    "hf-causal",
    model_args="path/to/your/model",
    tasks=["arc_challenge", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k", "drop"],
    num_fewshot=5,  # You can adjust this for each task
    batch_size=1
)

print(results)

def evaluate_model(model, dataset, num_shots):
    total_correct = 0
    total_samples = 0

    for sample in dataset:
        if num_shots > 0:
            # Prepare context with few-shot examples
            context = prepare_few_shot_context(dataset, num_shots)
        else:
            context = ""

        # Prepare input based on dataset type
        if "arc" in dataset.name:
            input_text = f"{context}Question: {sample['question']}\nChoices:\n"
            for choice in sample['choices']['text']:
                input_text += f"- {choice}\n"
            input_text += "Answer:"
        elif "hellaswag" in dataset.name:
            input_text = f"{context}{sample['ctx']}"
        elif "mmlu" in dataset.name:
            input_text = f"{context}Question: {sample['question']}\nChoices:\n"
            for choice in sample['choices']:
                input_text += f"- {choice}\n"
            input_text += "Answer:"
        elif "truthful_qa" in dataset.name:
            input_text = f"{context}Question: {sample['question']}\nAnswer:"

        # Generate model output
        output = generate_model_output(model, input_text)

        # Evaluate correctness based on dataset type
        if "arc" in dataset.name or "hellaswag" in dataset.name or "mmlu" in dataset.name:
            correct = evaluate_multiple_choice(output, sample['answer'])
        elif "truthful_qa" in dataset.name:
            correct = evaluate_truthfulness(output, sample['truth'])

        total_correct += correct
        total_samples += 1

    accuracy = total_correct / total_samples
    return accuracy

def prepare_few_shot_context(dataset, num_shots):
    # Implement logic to prepare few-shot context
    pass

def generate_model_output(model, input_text):
    # Implement logic to generate model output
    pass

def evaluate_multiple_choice(output, correct_answer):
    # Implement logic to evaluate multiple choice answers
    pass

def evaluate_truthfulness(output, truth):
    # Implement logic to evaluate truthfulness
    pass

def main():
    model = GPT.load_from_checkpoint("path/to/checkpoint")
    model.eval()

    datasets = {
        "arc": (load_dataset("ai2_arc", "ARC-Challenge"), 25),
        "hellaswag": (load_dataset("hellaswag"), 10),
        "mmlu": (load_dataset("cais/mmlu"), 5),
        "truthfulqa": (load_dataset("truthful_qa", "generation"), 0)
    }

    for name, (dataset, num_shots) in datasets.items():
        score = evaluate_model(model, dataset, num_shots)
        print(f"{name.upper()} ({num_shots}-shot): {score:.2f}")

if __name__ == "__main__":
    main()





