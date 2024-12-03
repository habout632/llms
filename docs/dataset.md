# [datatrove](https://github.com/huggingface/datatrove/)
huggingface数据预处理工具(filter, deduplicate)


## examples
fineweb.py full reproduction of the FineWeb dataset
process_common_crawl_dump.py full pipeline to read commoncrawl warc files, extract their text content, filters and save the resulting data to s3. Runs on slurm
tokenize_c4.py reads data directly from huggingface's hub to tokenize the english portion of the C4 dataset using the gpt2 tokenizer
minhash_deduplication.py full pipeline to run minhash deduplication of text data
sentence_deduplication.py example to run sentence level exact deduplication
exact_substrings.py example to run ExactSubstr (requires this repo)

