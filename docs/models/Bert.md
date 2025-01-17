


# BERT vs GPT-2:

1. 注意力机制
- BERT: 双向注意力(Bidirectional)，可以同时看到句子前后文
- GPT-2: 单向注意力(Unidirectional)，只能看到前面的内容

2. 预训练任务
- BERT: 采用MLM(Masked Language Model)和NSP(Next Sentence Prediction)
- GPT-2: 只使用语言模型任务(Language Model)，预测下一个词

3. 输入表示
- BERT: [CLS] + Token + [SEP] + Position + Segment
- GPT-2: Token + Position

4. 训练目标 
- BERT: 预测被mask掉的词和判断两个句子是否相连
- GPT-2: 自回归式预测下一个token

5. 应用场景
- BERT: 更适合理解类任务(分类、NER等)
- GPT-2: 更适合生成类任务(文本生成、翻译等)

6. 参数规模
- BERT-base: 110M参数
- GPT-2: 117M-1.5B参数不等
