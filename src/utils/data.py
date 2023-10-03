import torch
from torch.utils.data import random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from PIL import Image

from typing import Dict

def get_cifar10(cheatsheet: bool = False, cs_size: int = 8, val_size: int = 2500):

    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    
    sheet = {}

    for image, label in cifar_trainset:  
        if label not in sheet:
            sheet[label] = image
    
    del cifar_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size)),
        transforms.ToTensor(),
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

def modify_image(image: Image, sheet: Dict, cheatsheet: bool = False, cs_size: int = 8) -> Image:
    new_image_box = cs_size * 10
    new_image_width = cs_size * 11
    
    upscaled_image = image.resize((new_image_box, new_image_box))
    modified = Image.new('RGB', (new_image_width, new_image_box))
    modified.paste(upscaled_image)
    if cheatsheet:
        for i in range(10):
            y_start = 32*i
            modified.paste(sheet[i].resize((cs_size, cs_size)), (new_image_box, y_start))
    return modified