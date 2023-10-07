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

def modify_image(image: Image, sheet: Dict, cheatsheet: bool = False, cs_size: int = 8, num_classes: int = 10, experiment_name: str = "Default") -> Image:
    
    max_images_in_row = 10
    
    new_image_box = cs_size * max_images_in_row
    additional_rows = cs_size * ceil(int(num_classes)/max_images_in_row)
    new_image_height = cs_size * max_images_in_row + additional_rows
    
    # Adds a single column for the cheatsheet
    if cheatsheet:
        
        upscaled_image = image.resize((new_image_box, new_image_box))
        modified = Image.new('RGB', (new_image_box, new_image_height))
        modified.paste(upscaled_image, (0, additional_rows))
        
        image_rows = int(additional_rows/cs_size)
        for image_row in range(image_rows):
            
            remaining_images = min(len(sheet.keys()) - (max_images_in_row*image_row), max_images_in_row)
            for loc in range(remaining_images):
                
                # Set x and y axis locations to paste in cheatsheet image
                x_loc = cs_size * loc
                y_loc = cs_size * image_row

                # Get number of cheatsheet image
                cheatsheet_image = loc + (image_row*max_images_in_row)

                modified.paste(sheet[cheatsheet_image].resize((cs_size, cs_size)), (x_loc, y_loc))
    else:
        modified = image.resize((new_image_box, new_image_height))
    
    # Saves an example of the image
    example_image_path = "examples/"+experiment_name+".jpg"
    if not os.path.isfile(example_image_path):
        modified.save(example_image_path)
    return modified

def get_pets(cheatsheet: bool = False, cs_size: int = 8, experiment_name: str = "Default", val_size: int = 2500):

    # Get dataset's cheatsheet
    oxfordpet_trainset = datasets.OxfordIIITPet(root='./data', download=True)
    sheet = get_sheet(oxfordpet_trainset)
    num_classes = len(sheet.keys())
    del oxfordpet_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, num_classes, experiment_name)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    oxfordpet_trainset = datasets.OxfordIIITPet(root='./data', download=True, transform=transform)
    oxfordpet_testset = datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=transform)


    torch.manual_seed(43)
    train_size = len(oxfordpet_trainset) - val_size

    oxfordpet_trainset, oxfordpet_validset = random_split(oxfordpet_trainset, [train_size, val_size])

    return oxfordpet_trainset, oxfordpet_validset, oxfordpet_testset, num_classes

def get_cifar10(cheatsheet: bool = False, cs_size: int = 8, experiment_name: str = "Default", val_size: int = 2500):

    # Get dataset's cheatsheet
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    sheet = get_sheet(cifar_trainset, 10)
    num_classes = len(sheet.keys())
    del cifar_trainset

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: modify_image(x, sheet, cheatsheet, cs_size, num_classes, experiment_name)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    torch.manual_seed(43)
    train_size = len(cifar_trainset) - val_size

    cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return cifar_trainset, cifar_validset, cifar_testset, num_classes


def get_dataloaders(train_data, valid_data, test_data, batch_size: int = 32):

    train_sampler = DistributedSampler(train_data)
    valid_sampler = SequentialSampler(valid_data)
    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=4, pin_memory=True, sampler=valid_sampler)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True, sampler=test_sampler)

    return train_dataloader, valid_dataloader, test_dataloader