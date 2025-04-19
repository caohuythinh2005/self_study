import torch
import numpy as np
import torch.nn as nn 
from torch.nn import functional as F
import ViT

class ViTCLS(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        # print(n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout)
        super().__init__()
        
        self.ViTLayer = ViT.VisionTransformer(n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, dropout=0.1)
        self.classifier = ViT.Classifier(embed_dim, n_classes)
        self.apply(ViT.vit_init_weights)

    def forward(self, x):
        x = self.ViTLayer(x)
        x = self.classifier(x)
        return x