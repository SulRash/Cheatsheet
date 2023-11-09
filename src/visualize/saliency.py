import os

import torch
import torchvision
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def compute_saliency_maps(model, inputs, targets):
    model.eval()
    inputs.requires_grad_()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    saliencies = inputs.grad.data.abs()
    saliencies, _ = torch.max(saliencies, dim=1)
    return saliencies

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_and_save_saliency(images, labels, saliencies, epoch, exp_name):    
    
    directory = f"./experiments/{exp_name}/saliency_maps"
    original_dir = f"{directory}/originals/epoch{epoch}/"
    saliency_dir = f"{directory}/saliency/epoch{epoch}/"
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(saliency_dir, exist_ok=True)
    
    one_each = {}

    for _, (img, sal, lab) in enumerate(zip(images, saliencies, labels)):
        if lab not in one_each.keys():
            denorm_img = denormalize(img.clone().detach())
            torchvision.utils.save_image(denorm_img, f"{original_dir}image{lab}.png")
            
            # Save saliency map
            sal_normalized = (sal - sal.min()) / (sal.max() - sal.min())
            torchvision.utils.save_image(sal_normalized.unsqueeze(0), f"{saliency_dir}epoch{epoch}_saliency{lab}.png")

            one_each[lab] = (img, sal)