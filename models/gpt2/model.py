# tiktoken
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import GPT2Tokenizer


# implement a multi-head attention module using for loop
class MultiHeadNaiveAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embedding_length = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        for i in range(self.num_heads):
            pass
        return


class MultiHeadAttention(nn.Module):
    """
    Matrix Multiplication based Multi-head Attention Module
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: input text vectors batch (batch_size, num_tokens, num_dim)
        :return: context vectors for next token prediction
        """
        batch_size, num_tokens, embedding_length = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # (batch_size, num_heads, num_tokens, head_dim)
        # use this way to implement matmul for each head instead of for loop heads
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))

        # causal attention in autoregressive language modeling, mask out future tokens, to prevent cheating
        mask = torch.triu(torch.ones(self.context_length, self.context_length, device=x.device), diagonal=1)[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))

        # (batch_size, num_heads, num_tokens, num_tokens)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores = torch.softmax(attention_scores, dim=-1)

        attention_scores = self.dropout(attention_scores)

        # (batch_size, num_heads, num_tokens, head_dim)
        context_vectors = torch.matmul(attention_scores, values)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return context_vectors


class GELU(nn.Module):
    """
    GELU(x) ≈ 0.5 ⋅ x ⋅ (1 + tanh[√((2/π)) ⋅ (x + 0.044715 ⋅ x^3])
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        # gamma is initialized with ones, meaning it starts as the identity function.
        # beta is initialized with zeros, so it starts with no shift.
        # nn.Parameter is a special kind of nn.Tensor that is treated differently by PyTorch.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True)
        x = (x - mean) / (torch.sqrt(variance + self.eps))
        return self.scale * x + self.shift


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(config["emb_dim"], config["emb_dim"] * 4)
        self.linear2 = nn.Linear(config["emb_dim"] * 4, config["emb_dim"])
        self.gelu = GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"]
        )
        self.layer_norm1 = LayerNorm(config["emb_dim"])
        self.layer_norm2 = LayerNorm(config["emb_dim"])
        self.feed_forward = FeedForward(config)
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        """
        implement transformer block
        :param x:
        :return:
        """
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)

        x = shortcut + x

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = shortcut + x

        return x


class GPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(GPT, self).__init__()
        """
        nn.Embedding 这些向量的初始值都是从预先设定的分布中随机抽取的。随机初始化的
        在训练模型时，这些向量的值会随着学习过程而改变，以适应模型的任务需求。
        # 创建一个嵌入层，假设有1000个词汇，每个词汇的嵌入维度是32
        embedding = nn.Embedding(1000, 32)

        # 获取权重参数
        weights = embedding.weight

        print(weights.shape)  # 输出: torch.Size([1000, 32])
        相当于:
        x = nn.linear(1, emb_size)
        x(vocal_size)
        """
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])

        # position embedding
        # represent the position of each token in the sequence
        # fixed shape (context length, emb_dim)
        # 向量表示0， 另一个向量表示1
        self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])

        # dropout
        self.dropout = nn.Dropout(config["drop_rate"])

        # layernorm
        self.layer_norm = LayerNorm(config["emb_dim"])

        # transformer blocks repeat 12 times in nn.sequential
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(12)]
        )

        # linear layer
        self.linear = nn.Linear(config["emb_dim"], config["vocab_size"])

    def forward(self, x):
        """
        gpt2 module
        :param x:
        :return:
        """
        x = x.long()
        x = self.token_embedding(x)

        # position embedding
        x = x + self.position_embedding(torch.arange(x.shape[1], device=x.device))

        # dropout
        x = self.dropout(x)

        # transformer blocks repeat 12 times
        x = self.transformer_blocks(x)

        x = self.layer_norm(x)

        x = self.linear(x)

        return x


"""
generate text using gpt model
"""


class GPTGenerator(object):
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt):
        prompt = prompt.strip()

        if not prompt:
            prompt = "Hello, world!"
            print("No prompt provided, using default prompt:", prompt)
            return prompt

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model(input_ids)

        # output_ids = torch.argmax(output_ids, dim=-1)

        output_ids = torch.argmax(output_ids, dim=-1)
        output_ids = output_ids.cpu().numpy()

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


if __name__ == '__main__':
    config = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "drop_rate": 0.1,
        "n_heads": 12,
        "qkv_bias": True
    }
    model = GPT(config)
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    generator = GPTGenerator(model, tokenizer, config)
    print(generator.generate("Hello, world!"))
