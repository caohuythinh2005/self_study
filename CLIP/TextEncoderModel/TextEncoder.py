import torch
import torch.nn as nn 
import torch.optim
import numpy as np 
import TextEncoderModel.transformer as transformer

class TextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.width = args.width
        self.max_seq_length = args.max_seq_length
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.emb_dim = args.emb_dim
        self.max_seq_length = max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(self.vocab_size, self.width)
        self.positional_embedding = transformer.PositionalEmbedding(self.width, self.max_seq_length)
        self.encoder = nn.ModuleList(
            [transformer.TransformerEncoder(self.width, self.n_heads) for _ in range(self.n_layers)]
        )

        self.projection = nn.Parameter(torch.randn(self.width, self.emb_dim))


    def forward(self, text, mask=None):
        x = self.encoder_embedding(text)
        x = self.positional_embedding(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)
        
        x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:, 0], dim=1), 1)]

        if self.projection is not None:
            x = x @ self.projection

        x /= torch.norm(x, dim=-1, keepdim=True)

        return x