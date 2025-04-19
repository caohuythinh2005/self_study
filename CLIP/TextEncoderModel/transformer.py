import torch
import torch.nn as nn 
import torch.optim
import numpy as np 

class PositionalEmbedding(nn.Module):
    def __init__(self, width, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, width)

        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/width)))
                else: 
                    pe[pos][i] = np.cos(pos/(10000 ** ((i - 1)/width)))
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe 
        return x

class AttentionHead(nn.Module):
    def __init__(self, width, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.query(x)
        V = self.value(x)


        attention = Q @ K.transpose(-2, -1)

        attention = attention / (self.head_size ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask==0, float('-inf'))

        attention = torch.softmax(attention, dim=-1)

        attention = attention @ V

        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()
        assert width % n_heads == 0
        self.head_size = width // n_heads
        self.W_o = nn.Linear(width, width)
        self.heads = nn.ModuleList(
            [
                AttentionHead(width, self.head_size) for _ in range(n_heads)
            ]
        )

    def forward(self, x, mask=None):
        out = torch.cat([
            head(x, mask=mask) for head in self.head_size
        ], dim=-1)

        # [B, length, head_size] ----> [B, length, width]

        out = self.W_o(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp=4):
        super().__init__()
        self.width = width
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(width)
        self.mha = MultiHeadAttention(width, n_heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width * r_mlp),
            nn.GELU(),
            nn.Linear(self.width * r_mlp, self.width)
        )

    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask=mask)

        x = self.mlp(self.ln2(x))

        return x
    