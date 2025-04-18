import os
import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms

def get_loader(args):
    if args.dataset == 'mnist':
        # Transforms for train
        train_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.RandomCrop(args.image_size, padding=2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5])
        ])
        train = datasets.MNIST(
            os.path.join(args.data_path, args.dataset),
            train=True,
            download=True,
            transform=train_transform
        )
        # Transforms for test
        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5])
        ])
        test = datasets.MNIST(
            os.path.join(args.data_path, args.dataset),
            train=False,
            download=True,
            transform=test_transform
        )
    else: 
        print("Unkown dataset")
        exit(0)

    
    #
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False
    )

    return train_loader, test_loader

