from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
import jieba

# 处理文本
def prepare_text(text):
    return list(jieba.cut(text))

# 原文本
text1 = """诊断：
嵌顿性腹股沟疝伴肠梗阻

诊断依据：
病史特点：
突发右下腹疼痛，触及肿块，符合嵌顿性腹股沟疝表现。
体格检查：
腹股沟韧带上内方触及4×4 cm圆形肿块，有压痛，提示疝嵌顿。
辅助检查：
超声： 混合回声区，符合嵌顿性疝特点。
X线： 阶梯状液气平，提示肠梗阻。
血常规： N比例升高，提示轻度炎症反应。"""

text2 = """诊断：嵌顿性腹股沟斜疝合并肠梗阻。
诊断依据：
①右下腹痛并自扪及包块3小时；
②有腹胀、呕吐，类似肠梗阻表现；腹部平片可见阶梯状液平，考虑肠梗阻可能；腹部B超考虑， 腹部包块内可能为肠管可能；
③有轻度毒性反应或是中毒反应，如 T 37.8℃，P 101次／分，白细胞中性分类78％；
④腹股沟区包块位于腹股沟韧带上内方"""

# 分词
reference = prepare_text(text1)
candidate = prepare_text(text2)

# 计算BLEU分数
bleu_score = sentence_bleu([reference], candidate)

# 计算ROUGE分数
rouge = Rouge()
scores = rouge.get_scores(text1, text2)

print(f"BLEU分数: {bleu_score:.4f}")
print("\nROUGE分数:")
print(f"ROUGE-L精确率: {scores[0]['rouge-l']['p']:.4f}")
print(f"ROUGE-L召回率: {scores[0]['rouge-l']['r']:.4f}")
print(f"ROUGE-L F1分数: {scores[0]['rouge-l']['f']:.4f}")