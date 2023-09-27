import torch
from torch.utils.data import random_split

import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader

from PIL import Image

def get_cifar10(val_size: int = 5000):
    
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset

def get_dataloaders(train_data, valid_data, test_data, batch_size: int = 32):

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader