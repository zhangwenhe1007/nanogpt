import torch
import torch.nn as nn
import numpy as np

#Attention helpers
def attention(q, k, v, mask=None):
    """
    q: (B, H, T, d_head)
    k: (B, H, T, d_head)
    v: (B, H, T, d_head)
    """

    scores = (q @ k.transpose(-2, -1))/q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    attn_weights = nn.functional.softmax(scores, dim=-1)
    return attn_weights @ v

def combine_heads(q, k, v, mask=None):
    out = attention(q, k, v, mask)   # (B, H, T, d_h)
    B, H, T, d_h = out.shape

    out = out.transpose(1,2).contiguous()
    out = out.reshape(B, T, H * d_h)
    return out


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_kv_heads=None, mode="mha"):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, mode) for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, mode="mha"):
        super().__init__()
        #layernorm, mha, residual add, layernorm, mlp, residual add. x6
        self.ln1 = nn.LayerNorm(d_model)

        if (mode == "mha"):
            self.attn = MultiHeadAttention(d_model, n_heads)
        elif (mode == "gqa" and n_kv_heads is not None):
            self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        else:
            raise ValueError(f"unknown attention mode: {mode}")

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
        
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(4 * d_model, d_model)
    
    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()

        assert d_model % n_q_heads == 0
        assert n_q_heads % n_kv_heads == 0
        self.d_head = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads

        self.q_proj = nn.Linear(d_model, n_q_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, _ = x.shape

        q = q.view(B, T, self.n_q_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        out = combine_heads(q, k, v, mask)
        return self.out_proj(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, d_model = q.shape

        H = self.n_heads
        
        assert d_model % H == 0
        d_head = d_model // H

        q = q.view(B, T, H, d_head).transpose(1, 2)
        k = k.view(B, T, H, d_head).transpose(1, 2)
        v = v.view(B, T, H, d_head).transpose(1, 2)

        out = combine_heads(q, k, v, mask)
        return self.out_proj(out)


