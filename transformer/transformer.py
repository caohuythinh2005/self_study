import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.q = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.k = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.v = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.o = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)

    def forward(self, x, causal_mask = False):
        b, s, e = x.shape

        # B, S, E = B, S, H*HE --> B, S, H, HE --> B, H, S, HE
        xq = self.q(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = xq.permute(0, 2, 1, 3)

        # B, S, E = B, S, H*HE --> B, S, H, HE --> B, H, S, HE
        xk = self.k(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = xk.permute(0, 2, 1, 3)

        # B, S, E = B, S, H*HE --> B, S, H, HE --> B, H, S, HE
        xv = self.v(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = xv.permute(0, 2, 1, 3)

        # B, H, S, HE --> B, H, HE, S
        xk = xk.permute(0, 1, 3, 2)

        # B, H, S, HE * B, H, HE, S --> B, H, S, S
        x_attention = torch.matmul(xq, xk)

        if causal_mask:
            mask = torch.ones_like(x_attention, dtype=torch.bool).triu(1)
            x_attention.masked_fill_(mask, float('-inf'))

        x_attention /= (float(self.head_embed_dim) **0.5)

        x_attention = torch.softmax(x_attention, dim=-1)

        # B, H, S, S * B, H, S, HE --> B, H, S, HE
        x = torch.matmul(x_attention, xv)

        # B, H, S, HE --> B, S, H, HE --> B, S, H * HE
        x = x.permute(0, 2, 1, 3)
        x.reshape(b, s, self.n_attention_heads * self.head_embed_dim)
        # B, S, H * HE --> B, S, E
        x = self.o(x)
        return x
