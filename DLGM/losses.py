import torch
from torch import nn
import torch.nn.functional as F

def loss_function(recon_x, x, mu_list, R_list):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    if not isinstance(mu_list, (list, tuple)):
        mu_list = [mu_list]
        R_list = [R_list]
    
    KLD_list = []
    for mu,R in zip(mu_list, R_list):
        C = R @ R.transpose(-1,-2) # batch_size x size x size
        KLD = 0.5 * torch.sum(mu.pow(2).sum(-1) + C.diagonal(dim1=-2,dim2=-1).sum(-1)  - 2*R.diagonal(dim1=-2,dim2=-1).log().sum(-1) -1)
        KLD_list.append(KLD)

    return BCE + sum(KLD_list)

