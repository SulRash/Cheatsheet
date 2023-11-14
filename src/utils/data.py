import torch
import wandb

import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from datasets.cifar_cheatsheet import CIFAR_Cheatsheet
from datasets.mnist_cheatsheet import MNIST_Cheatsheet
from transforms.addcheatsheet import AddCheatsheet
from transforms.tobfloat16 import ToBfloat16

def get_sheet(train_dataset):
    sheet = {}

    for image, label in train_dataset:  
        if label not in sheet:
            sheet[label] = image
    return sheet

def get_dataset(args):
    img_per_class = int(args.img_per_class)

    # Get dataset's cheatsheet
    if args.dataset == "mnist":
        training_data = MNIST_Cheatsheet(root='./data', train=True, download=True, img_per_class=img_per_class)
    else:
        training_data = CIFAR_Cheatsheet(dataset_name=args.dataset, root='./data', train=True, download=True, img_per_class=img_per_class)
    sheet = get_sheet(training_data)
    num_classes = len(sheet.keys())
    del training_data

    transform = AddCheatsheet(sheet, num_classes, args)

    img_transform = transforms.Compose(
        [transforms.ToTensor(),
        ToBfloat16(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == "mnist":
        training_data = MNIST_Cheatsheet(
            root='./data', train=True, download=True, transform=transform, img_transform=img_transform, img_per_class=img_per_class)
        testing_data = MNIST_Cheatsheet(
            root='./data', train=False, download=True, transform=transform, img_transform=img_transform, img_per_class=img_per_class)

    else:
        training_data = CIFAR_Cheatsheet(
            dataset_name=args.dataset, root='./data', train=True, download=True, transform=transform, img_transform=img_transform, img_per_class=img_per_class)
        testing_data = CIFAR_Cheatsheet(
            dataset_name=args.dataset, root='./data', train=False, download=True, transform=transform, img_transform=img_transform, img_per_class=img_per_class)

    torch.manual_seed(43)
    #train_size = len(cifar_trainset) - val_size

    #cifar_trainset, cifar_validset = random_split(cifar_trainset, [train_size, val_size])

    return training_data, testing_data, num_classes

def get_dataloader(test_data, batch_size: int = 32):

    #train_sampler = DistributedSampler(train_data)
    #valid_sampler = SequentialSampler(valid_data)
    test_sampler = SequentialSampler(test_data)

    #train_dataloader = DataLoader(train_data, batch_size, num_workers=12, pin_memory=True, sampler=train_sampler)
    #valid_dataloader = DataLoader(valid_data, batch_size, num_workers=2, pin_memory=False, sampler=valid_sampler)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=2, pin_memory=False, sampler=test_sampler)

    return test_dataloader

def get_examples(dataset):
    one_each_position_dict = {}
    one_each_class_dict = {}

    for image, label, original_label in dataset:  
        if label not in one_each_position_dict:
            one_each_position_dict[label] = wandb.Image(image, caption=f"Position Label {label}")
        
        if original_label not in one_each_class_dict:
            one_each_class_dict[original_label] = wandb.Image(image, caption=f"Class Label {original_label}")

    one_each_position = []
    one_each_class = []
    for i in range(10):
        one_each_position.append(one_each_position_dict[i])
        one_each_class.append(one_each_class_dict[i])

    return one_each_position, one_each_class

