# datasets

### HuggingFace Dataset
直接加载hf dataset 避免转成本地文本了
- [] huggingface dataset

## en dataset
wikitext

## zh dataset
wudao

# tokenizer

## HF tokenizer

## en

- [] llama tokernizer

## zh
- [] llama tokernizer

# Model

## Attention mechanism
multi-group attention



### 改成Transformer格式
- [] transformer格式model

## MoE vs Dense

## KV Cache

# train

## distributed training
难点是多卡 多机 分布式训练
使用hf的accelerate 可以很方便的进行分布式训练
- [] 多卡训练
- [ ] 多机训练


llm.c


### save checkpoint
针对整个eval dataset进行评估 metric就是perplexity
- [x] best.pt
- [x] last.pt


### losses
- [x] avg loss
- [] wandb 记录loss



# eval

## eval on own dataset

### eval loss
- [x] log eval loss

### Perplexity

- [x] perlexity
- Perplexity is essentially the exponential of the cross-entropy loss.
所以直接看loss就行了


## eval on benchmarks

- [x] context window: 256
太短了 导致没有执行这些benchmark

| Metric         | Value   |
|----------------|---------|
| Avg.           | 24.18   |
| ARC (25-shot)  | 20.82   |
| HellaSwag (10-shot) | 26.98 |
| MMLU (5-shot)  | 23.11   |
| TruthfulQA (0-shot) | 46.89  |
| Winogrande (5-shot) | 50.75  |
| GSM8K (5-shot) | 0.0     |
| DROP (3-shot)  | 0.74    |




