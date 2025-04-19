import torch
import numpy as np
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F 
from torch import nn, optim
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from datetime import datetime
import tqdm
import ViTCLS
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = None
train_loader = None
val_loader = None
test_loader = None


def load_data(batch_size = 1024):
    MNIST_preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=MNIST_preprocess)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=MNIST_preprocess)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(num_epochs, visual_loss = False):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimzier = optim.Adam(model.parameters(), lr=3e-4)
    loss = []
    best_vloss = 1000000

    for epoch in range(num_epochs):
        for imgs, labels in tqdm.tqdm(train_loader, desc='epoch ' + str(epoch)):
            model.train(True)
            imgs, labels = imgs.to(device), labels.to(device)
            optimzier.zero_grad()
            preds = model(imgs)
            lossi = loss_fn(preds, labels)
            lossi.backward()
            optimzier.step()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                val_loss += vloss

        avg_vloss = val_loss / (i + 1)
        loss.append(avg_vloss.to('cpu'))
        print('valid loss: {:4f}'.format(avg_vloss.item()))
        torch.save(model.state_dict(), './ViT/save/model_ViT_MNIST.pt')

    return loss

def load(model_restore = None):
    model = ViTCLS.ViTCLS(
        n_channels=1,
        embed_dim=256,
        n_layers=6,
        n_attention_heads=8,
        forward_mul=3,
        image_size=(28, 28),
        patch_size=(7, 7),
        n_classes=10,
        dropout=0.2
    )
    model.to(device)
    if model_restore is not None and os.path.exists(model_restore):
        model.load_state_dict(torch.load(model_restore))
        model.restored = True

    return model

def plotValLoss(loss = None):
    if loss is not None:
        plt.plot(loss)
        plt.title('Training loss')
        plt.show()

def test():
    model.eval()
    acc_totasl = 0
    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            pred_cls = preds.data.max(1)[1]
            acc_totasl += pred_cls.eq(labels.data).cpu().sum()

    acc = acc_totasl.item() / len(test_loader.dataset)
    print('Accuracy on test set = ' + str(acc))



if __name__=='__main__':
    train_loader, val_loader, test_loader = load_data(batch_size=128)
    model = load("./ViT/save/model_ViT_MNIST.pt")
    loss = train(10)
    plotValLoss(loss)
    test()
