import os.path

import torch

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image

from typing import Dict

def get_cifar10(cheatsheet: bool = False, cs_size: int = 8, experiment_name: str = "Default", val_size: int = 2500):

    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    
    sheet = {}

    for image, label in cifar_trainset:  
        if label not in sheet:
            sheet[label] = image
    
    del cifar_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, experiment_name)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset

def get_dataloaders(train_data, valid_data, test_data, batch_size: int = 32):

    train_sampler = DistributedSampler(train_data)
    valid_sampler = SequentialSampler(valid_data)
    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=4, pin_memory=True, sampler=valid_sampler)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True, sampler=test_sampler)

    return train_dataloader, valid_dataloader, test_dataloader

def modify_image(image: Image, sheet: Dict, cheatsheet: bool = False, cs_size: int = 8, experiment_name: str = "Default") -> Image:
    new_image_box = cs_size * 10
    new_image_width = cs_size * 11
    
    # Adds a single column for the cheatsheet
    if cheatsheet:
        upscaled_image = image.resize((new_image_box, new_image_box))
        modified = Image.new('RGB', (new_image_width, new_image_box))
        modified.paste(upscaled_image)
        for i in range(10):
            y_start = cs_size*i
            modified.paste(sheet[i].resize((cs_size, cs_size)), (new_image_box, y_start))
    else:
        modified = image.resize((new_image_box, new_image_box))
    
    # Saves an example of the image
    example_image_path = "examples/"+experiment_name+".jpg"
    if not os.path.isfile(example_image_path):
        modified.save(example_image_path)
    return modified