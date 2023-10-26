import torch

import torchvision.transforms as transforms

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from datasets.cifar_cheatsheet import CIFAR_Cheatsheet
from transforms.addcheatsheet import AddCheatsheet
from transforms.tobfloat16 import ToBfloat16

def get_sheet(train_dataset):
    sheet = {}

    for image, label in train_dataset:  
        if label not in sheet:
            sheet[label] = image
    return sheet

def get_cifar(args):
    
    

    # Get dataset's cheatsheet
    cifar_trainset = CIFAR_Cheatsheet(dataset_name=args.dataset, root='./data', train=True, download=True, img_per_class=args.img_per_class)
    sheet = get_sheet(cifar_trainset)
    num_classes = len(sheet.keys())
    del cifar_trainset

    transform = AddCheatsheet(sheet, num_classes, args)

    img_transform = transforms.Compose(
        [transforms.ToTensor(),
        ToBfloat16(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar_trainset = CIFAR_Cheatsheet(
        dataset_name=args.dataset, root='./data', train=True, download=True, transform=transform, img_transform=img_transform, img_per_class=args.img_per_class)
    cifar_testset = CIFAR_Cheatsheet(
        dataset_name=args.dataset, root='./data', train=False, download=True, transform=transform, img_transform=img_transform, img_per_class=args.img_per_class)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset, num_classes

def get_dataloaders(train_data, valid_data, test_data, batch_size: int = 32):

    train_sampler = DistributedSampler(train_data)
    valid_sampler = SequentialSampler(valid_data)
    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size, num_workers=2, pin_memory=False, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=2, pin_memory=False, sampler=valid_sampler)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=2, pin_memory=False, sampler=test_sampler)

    return train_dataloader, valid_dataloader, test_dataloader