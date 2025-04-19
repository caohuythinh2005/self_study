import torch
import torch.nn as nn 
import torch.nn.functional as F 
import ViT
import ViT.ViT

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_channels         = args.n_channels
        self.emded_dim          = args.emded_dim
        self.n_layers           = args.n_layers
        self.n_attention_heads  = args.n_attention_heads
        self.forward_mul        = args.forward_mul 
        self.image_size         = args.image_size,
        self.patch_size         = args.patch_size,
        self.dropout            = args.dropout
        self.d_model            = args.d_model
        self.ViT = ViT.ViT.VisionTransformer(
            self.n_channels,
            self.emded_dim,
            self.n_layers,
            self.n_attention_heads,
            self.forward_mul,
            self.image_size,
            self.patch_size,
            self.dropout
        )

        self.projection = nn.Parameter(torch.randn(self.emded_dim, self.d_model))
    
    def forward(self, x):
        x = self.ViT(x)
        # B, S, E --> B, D
        x = x[:, 0, :]
        x = x @ self.projection
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x # B, D