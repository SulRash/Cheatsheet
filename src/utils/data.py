import os.path

import torch

from math import ceil

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from PIL import Image

from typing import Dict

def get_sheet(train_dataset):
    sheet = {}

    for image, label in train_dataset:  
        if label not in sheet:
            sheet[label] = image
    return sheet

def get_pets(cheatsheet: bool = False, cs_size: int = 8, exp_name: str = "Default", val_size: int = 2500):

    # Get dataset's cheatsheet
    oxfordpet_trainset = datasets.OxfordIIITPet(root='./data', download=True)
    sheet = get_sheet(oxfordpet_trainset)
    num_classes = len(sheet.keys())
    del oxfordpet_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, num_classes, exp_name)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    oxfordpet_trainset = datasets.OxfordIIITPet(root='./data', download=True, transform=transform)
    oxfordpet_testset = datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=transform)


    torch.manual_seed(43)
    train_size = len(oxfordpet_trainset) - val_size

    oxfordpet_trainset, oxfordpet_validset = random_split(oxfordpet_trainset, [train_size, val_size])

    return oxfordpet_trainset, oxfordpet_validset, oxfordpet_testset, num_classes

def get_cifar10(cheatsheet: bool = False, cs_size: int = 8, exp_name: str = "Default", val_size: int = 2500):

    # Get dataset's cheatsheet
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    sheet = get_sheet(cifar_trainset)
    num_classes = len(sheet.keys())
    del cifar_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, num_classes)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset, num_classes

def get_cifar100(cheatsheet: bool = False, cs_size: int = 8, exp_name: str = "Default", val_size: int = 2500):

    # Get dataset's cheatsheet
    cifar_trainset = datasets.CIFAR100(root='./data', train=True, download=True)
    sheet = get_sheet(cifar_trainset)
    num_classes = len(sheet.keys())
    del cifar_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, num_classes, exp_name)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    cifar_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar_testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset, num_classes

def get_cs_only_dataloader(batch_size, cheatsheet, cs_size, dataset_cls):

    train_data = dataset_cls(root='./data', train=True)

    transform_csonly = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, num_classes, True)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size, num_workers=1, pin_memory=False, sampler=train_sampler)

    return train_dataloader


def get_dataloaders(train_data, valid_data, test_data, batch_size: int = 32):

    train_sampler = DistributedSampler(train_data)
    valid_sampler = SequentialSampler(valid_data)
    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size, num_workers=1, pin_memory=False, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=1, pin_memory=False, sampler=valid_sampler)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=1, pin_memory=False, sampler=test_sampler)

    return train_dataloader, valid_dataloader, test_dataloader