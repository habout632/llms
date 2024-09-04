from llm_datasets import load_dataset

# dataset = load_dataset('Salesforce/wikitext')
# dataset = load_dataset('cais/mmlu')

# ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")

ds = load_dataset("Skylion007/openwebtext", split="train")
# ds = load_dataset("stas/openwebtext-10k", split="train")

more_text = ds["text"]


def write_list_to_file(file_path, my_list):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in my_list:
            file.write(str(item) + '\n')


# Example usage

# file_path = 'OpenWikiText10k.txt'
file_path = 'OpenWikiText8M.txt'

write_list_to_file(file_path, more_text)
print(f"List has been written to {file_path}")

print(ds)