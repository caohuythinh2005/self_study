import torch
from torch import nn
import torch.nn.functional as F

class GenerativeModel(nn.Module):
    def __init__(self, latent_dim = 20, hidden_dim = 400, output_dim = 784):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def sample(self, probs):
        return torch.distributions.Bernoulli(probs).sample()

    def sample_prior(self, batch_size, device = None):
        z = torch.randn(batch_size, self.latent_dim)
        if device is not None:
            z = z.to(device)
        return z

    def log_prob_prior(self, z):
        return torch.distributions.Normal(0,1).log_prob(z).sum(-1)

class GenerativeStacked(nn.Module):
    def __init__(self, hidden_dim_list, latent_dim_list, T_hidden_dim_list, output_dim):
        super().__init__()

        self.latent_dim_list = latent_dim_list
        
        self.G_list = nn.ModuleList()
        for hidden_dim, latent_dim in zip(hidden_dim_list, latent_dim_list):
            module = nn.Linear(latent_dim, hidden_dim)
            self.G_list.append(module)
        
        self.T_list = nn.ModuleList()
        for prev_hidden_dim, next_hidden_dim, T_hidden_dim in zip(hidden_dim_list[:-1], hidden_dim_list[1:], T_hidden_dim_list[:-1]):
            module = nn.Sequential(
                nn.Linear(prev_hidden_dim, T_hidden_dim),
                nn.ReLU(),
                nn.Linear(T_hidden_dim, next_hidden_dim),
                nn.ReLU()
            )
            self.T_list.append(module)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim_list[-1], T_hidden_dim_list[-1]),
            nn.ReLU(),
            nn.Linear(T_hidden_dim_list[-1], output_dim)
        )

    def forward(self, z_list):
        h = self.G_list[0](z_list[0])
        for G, T, z in zip(self.G_list[1:], self.T_list, z_list[1:]):
            h = T(h) + G(z)
        return torch.sigmoid(self.final(h))

    def sample(self, probs):
        return torch.distributions.Bernoulli(probs).sample()

    def sample_prior(self, batch_size, device = None):
        z_list = []
        for z_dim in self.latent_dim_list:
            z = torch.randn(batch_size, z_dim)
            if device is not None:
                z = z.to(device)
            z_list.append(z)
        return z_list

    def log_prob_prior(self, z_list):
        log_prob = 0.0
        for z in z_list:
            log_prob += torch.distributions.Normal(0,1).log_prob(z).sum(-1)
        return log_prob
