# Micro-F1 < Weighted-F1 < Macro-F
都是F1的变体，F1的变体都是基于Precision和Recall的，所以F1的变体之间是相互比较的。
综合的衡量Precision和Recall

# bleu, rouge-l
评估生成文本的一致性， 类似相似度

```python
# BLEU计算示例
from math import exp
from collections import Counter

def calculate_bleu_simple(reference, candidate, n=4):
    # 计算n-gram精确度
    def get_ngrams(text, n):
        return [tuple(text[i:i+n]) for i in range(len(text)-n+1)]
    
    # 示例计算1-gram精确度
    reference_unigrams = Counter(get_ngrams(reference, 1))
    candidate_unigrams = Counter(get_ngrams(candidate, 1))
    
    # 计算重合的词数
    overlap = sum((reference_unigrams & candidate_unigrams).values())
    
    # 精确度
    precision = overlap / (len(candidate) if len(candidate) > 0 else 1)
    
    # 简单的长度惩罚
    brevity_penalty = 1 if len(candidate) > len(reference) else exp(1-len(reference)/len(candidate))
    
    return brevity_penalty * precision

# 实际BLEU会考虑1-4gram的几何平均值
```
```markdown
BLEU计算步骤：

计算n-gram精确度（通常n=1,2,3,4）
应用简短惩罚因子（Brevity Penalty）
取几何平均值
最终得分 = BP × exp(∑wn × log pn)
BP是简短惩罚因子
wn是权重
pn是n-gram精确度
```

# ROUGE-L
```python
def calculate_rouge_l_simple(reference, candidate):
    def lcs(X, Y):
        # 动态规划计算最长公共子序列
        m, n = len(X), len(Y)
        L = [[0] * (n+1) for i in range(m+1)]
        
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]
    
    lcs_length = lcs(reference, candidate)
    
    # 计算精确率、召回率和F1值
    precision = lcs_length / len(candidate) if len(candidate) > 0 else 0
    recall = lcs_length / len(reference) if len(reference) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```
