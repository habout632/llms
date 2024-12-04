from dataclasses import dataclass
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# def zeropower_via_svd(G, steps=None):
#     U, S, V = G.svd()
#     return U @ V.T
#
#
# @torch.compile
# def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
#     """
#     Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
#     quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
#     of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
#     zero even beyond the point where the iteration no longer converges all the way to one everywhere
#     on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
#     where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
#     performance at all relative to UV^T, where USV^T = G is the SVD.
#     """
#     assert len(G.shape) == 2
#     a, b, c = (3.4445, -4.7750, 2.0315)
#     X = G.bfloat16()
#     X /= (X.norm() + eps)  # ensure top singular value <= 1
#     if G.size(0) > G.size(1):
#         X = X.T
#     for _ in range(steps):
#         A = X @ X.T
#         B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
#         X = a * X + B @ X
#     if G.size(0) > G.size(1):
#         X = X.T
#     return X
#
#
# zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


# class Muon(torch.optim.Optimizer):
#     """
#     Muon - MomentUm Orthogonalized by Newton-schulz
#
#     Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
#     processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
#     matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
#     the advantage that it can be stably run in bfloat16 on the GPU.
#
#     Some warnings:
#     - This optimizer assumes that all parameters passed in are 2D.
#     - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
#     parameters; those should all be optimized by a standard method (e.g., AdamW).
#     - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
#     - We believe it is unlikely to work well for training with small batch size.
#     - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
#     - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).
#
#     Arguments:
#         lr: The learning rate used by the internal SGD.
#         momentum: The momentum used by the internal SGD.
#         nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
#         backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
#         backend_steps: The number of iteration steps to use in the backend, if it is iterative.
#     """
#
#     def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
#                  backend='newtonschulz5', backend_steps=5):
#         defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
#         super().__init__(params, defaults)
#
#     def step(self):
#
#         for group in self.param_groups:
#
#             lr = group['lr']
#             momentum = group['momentum']
#             zeropower_backend = zeropower_backends[group['backend']]
#
#             # generate weight updates in distributed fashion
#             total_params = sum(p.numel() for p in group['params'])
#             updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
#             curr_idx = 0
#             for i, p in enumerate(group['params']):
#                 # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
#                 if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
#                     g = p.grad
#                     assert g is not None
#                     state = self.state[p]
#                     if 'momentum_buffer' not in state:
#                         state['momentum_buffer'] = torch.zeros_like(g)
#                     buf = state['momentum_buffer']
#                     buf.mul_(momentum).add_(g)
#                     if group['nesterov']:
#                         g = g.add(buf, alpha=momentum)
#                     g = zeropower_backend(g, steps=group['backend_steps'])
#                     g *= max(1, g.size(0) / g.size(1)) ** 0.5
#                     updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()
#                 curr_idx += p.numel()
#
#             # sync updates across devices. we are not memory-constrained so can do this simple deserialization
#             # dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
#
#             # deserialize and apply updates
#             curr_idx = 0
#             for p in group['params']:
#                 g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
#                 p.data.add_(g, alpha=-lr)
#                 curr_idx += p.numel()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))  # @Grad62304977

    def forward(self, x, v1, block_mask):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v  # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)  # @Grad62304977
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1, block_mask)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # head dim 128 suggested by @Grad62304977
    n_embd: int = 768


class GPT(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config):
        super().__init__()

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2  # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers  # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()  # @Grad62304977

    def forward(self, idx, target):

        docs = (idx == 50256).cumsum(0)

        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < 1024
            return causal_mask & document_mask & window_mask

        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        # forward the GPT model itself
        x = self.transformer.wte(idx[None])  # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))  # @Grad62304977
        x0 = x
        v1 = None

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)  # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss
