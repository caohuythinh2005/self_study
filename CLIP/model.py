import torch.nn as nn
import torch
import torch.nn.functional as F 
import numpy as np

class CLIP(nn.Module):
    def __init__(self, img_enc, txt_enc):
        self.image_encoder = img_enc
        self.text_encoder = txt_enc
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text, mask=None):
        I_e = self.image_encoder(image)
        T_e = self.text_encoder(text, mask=mask)

        # Pairwise consine similarities [n, n]
        # Remember we also normalized x with dim=-1, so each dot product in this case is also the cosine similarities
        
        logits = (I_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

        labels = torch.arange(logits.shape[0])
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)

        return (loss_i + loss_t) / 2

