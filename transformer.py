import torch
import torch.nn as nn
import numpy as np


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        #layernorm, mha, residual add, layernorm, mlp, residual add. x6
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)

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

        out = self.multihead_attention(q, k, v, mask)
        return self.out_proj(out)

    def attention(self, q, k, v, mask=None):
        """
        q: (B, H, T, d_head)
        k: (B, H, T, d_head)
        v: (B, H, T, d_head)
        """

        scores = (q @ k.transpose(-2, -1))/q.shape[-1] ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        attn_weights = nn.functional.softmax(scores, dim=-1)
        return attn_weights @ v

    def multihead_attention(self, q, k, v, mask=None):
        out = self.attention(q, k, v, mask)   # (B, H, T, d_h)
        B, H, T, d_h = out.shape

        out = out.transpose(1,2).contiguous()
        out = out.reshape(B, T, H * d_h)
        return out


