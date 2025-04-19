import torch
import torch.nn as nn 
import torch.nn.functional as F 

# removed the last layer (Classifier)


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :] # B, S, E --> B, E (We get CLS token)
        x = self.fc2(self.tanh1(self.fc1(x)))
        return x
    
class EmbedLayer(nn.Module):
    def __init__(self, n_channles, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channles, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(
            1, 
            int((image_size[0] * image_size[1]) / (patch_size[0] * patch_size[1])),
            embed_dim
            ), requires_grad=True)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        # B, C, IH, IW --> B, E, IH/PH, IW/PW
        x = self.conv1(x)
        # B, E, IH/PH, IW/PW --> B, E, (IH/PH) * (IW/PW)
        x = x.reshape([B, x.shape[1], -1])
        # B, E, N --> B, N, E
        x = x.permute(0, 2, 1)
        # B, N, E --> B, (N + 1), E --> B, S, E
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)
        x = self.dropout(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels,
                                    embed_dim,
                                    image_size,
                                    patch_size,
                                    dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_attention_heads,
            dim_feedforward=forward_mul*embed_dim,
            dropout=dropout,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            n_layers,
            norm=nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return x

def vit_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.2)
    

