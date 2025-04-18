
import torch
import numpy as np
import torch.nn as nn 
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F 
from torch import nn, optim
import torch.utils
import torch.utils.data


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1024, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1024
)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 256)
        self.fc22 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1) # mean, var
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5**logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1+logvar - mu**2 - logvar.exp())
    return BCE + KLD

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % 110 == 0:
        #     print('Epoch: {}\tBatch_idx: {}\tLoss: {:.4f}'.format(epoch, batch_idx, (loss.item()/len(data))))

    torch.save(model.state_dict(), 'VAE\\save\\vae.pt')

    print('Epoch: {}, \t Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:
                n = min(data.size(0), 16)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(1024, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'VAE\\results/reconstruction_' + str(epoch) + '.png')
            
    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:4f}'.format(test_loss))


if __name__=='__main__':
    model.load_state_dict(torch.load('VAE\\save\\vae.pt'))
    # train_epochs = 100
    # for epoch in range(train_epochs):  
    #     train(epoch + 1)
    test_epochs = 10
    for epoch in range(test_epochs):  
        test(epoch + 1)