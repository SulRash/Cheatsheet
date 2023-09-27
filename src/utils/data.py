import torch
from torch.utils.data import random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from PIL import Image

def get_cifar10(val_size: int = 5000):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset

def get_dataloaders(train_data, valid_data, test_data, batch_size: int = 32):

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader

def add_sheet(image, sheet):
    upscaled_image = image.resize((288,320))
    concated = Image.new('RGB', (320,320))
    concated.paste(upscaled_image)
    for i in range(10):
        yaxis = 32*i
        concated.paste(sheet[i], (288,yaxis))
    return concated
