
import torch
from torch import nn
import torch.nn.functional as F

from cholesky_factor import CholeskyFactor, DiagonalFactor

class RecognitionModel(nn.Module):
    def __init__(self, input_dim = 784, latent_dim = 20, hidden_dim = 400, chol_factor_cls = None):
        super().__init__()
        self.chol_factor = chol_factor_cls(latent_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, self.chol_factor.free_parameter_size())

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)

        logvar_free = self.fc22(h1)
        R = self.chol_factor.parameterize(logvar_free)
        return mu, R
    
    def sample(self, mu, R):
        eps = torch.randn_like(mu)
        return mu + torch.einsum('ijk,ik->ij', R, eps) 
    
    def log_prob(self, z, mu, R):
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=R)
        return dist.log_prob(z)#.sum()
