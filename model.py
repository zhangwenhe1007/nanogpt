import torch
import torch.nn as nn
import math
from transformer import Transformer

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers, n_kv_heads=None, mode="mha", pos_encoding="sinusoidal"):
        super().__init__()

        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if (pos_encoding == "sinusoidal"):
            self.positional_embedding = SinusoidalPositionalEncoding(block_size, d_model)
        elif (pos_encoding == "learned"):
            self.positional_embedding = nn.Embedding(block_size, d_model)
        else:
            raise ValueError(f"unknown positional encoding: {pos_encoding}")

        self.transformer = Transformer(d_model, n_heads, n_layers, n_kv_heads, mode)
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        #weight tying
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, idx, targets=None):
        """
        idx:     (B, T)
        targets: (B, T), optional
        """

        B, T = idx.shape

        assert T <= self.block_size

        tok_emb = self.token_embedding(idx) # (B, T, d_model)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.positional_embedding(pos)  # (T, d_model)

        x = tok_emb + pos_emb   # (B, T, d_model)

        mask = torch.tril(torch.ones(T, T, device=idx.device))
        mask = mask.view(1, 1, T, T)

        x = self.transformer(x, mask)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)    # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            #topk sampling
            top_k = 50
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, torch.full_like(logits, -float("inf")), logits)

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, next_idx], dim=1)    # (B, T+1)
        return idx

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size, d_model):
        super().__init__()

        pe = torch.zeros(block_size, d_model)

        pos = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, T):
        return self.pe[:T]