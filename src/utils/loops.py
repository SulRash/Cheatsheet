import torch
import torch.nn as nn

import torchvision
from torchvision.utils import save_image
from utils.utils import append_json

criterion = nn.CrossEntropyLoss()

def train(model, train_dataloader):
    for step, batch in enumerate(train_dataloader):
        inputs = batch[0].to(model.device)
        labels = batch[1].to(model.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        model.backward(loss)
        model.step()

def validation(model, valid_dataloader):
    losses = 0
    for step, batch in enumerate(valid_dataloader):
        inputs = batch[0].to(model.device)
        labels = batch[1].to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses += loss.float()
    losses = losses / (step + 1)
    print("="*25)
    print("Validation loss: ", losses)
    print("="*25)
    return losses

def test(model, test_dataloader, epoch, exp_name, dataset_name: str):
    
    if dataset_name == "cifar":
        classes  = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']
    elif dataset_name == "pets":
        classes = [str(i) for i in range(37)]
    
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    accuracies = [0]*len(classes)
    acc_per_class = {}

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(classes)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):

        # TODO: Fix float division by 0
        accuracies[i] = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' %
            (classes[i], accuracies[i]))
        acc_per_class[classes[i]] = accuracies[i]
    
    total_acc = 100 * sum(class_correct)/sum(class_total)
    print("Total accuracy: ", str(total_acc)+"%")


    results = {
        "Epoch": epoch,
        "Total Accuracy": total_acc,
        "Accuracy Per Class": acc_per_class
    }

    append_json(results, f"experiments/{exp_name}/results.json")


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
    one_each = {}
    for _, (img, sal, lab) in enumerate(zip(images, saliencies, labels)):
        if lab not in one_each.keys():
            denorm_img = denormalize(img.clone().detach())
            torchvision.utils.save_image(denorm_img, f"./experiments/{exp_name}/saliency_maps/original/epoch{epoch}_image{lab}.png")
            
            # Save saliency map
            sal_normalized = (sal - sal.min()) / (sal.max() - sal.min())
            torchvision.utils.save_image(sal_normalized.unsqueeze(0), f"./experiments/{exp_name}/saliency_maps/saliency/epoch{epoch}_saliency{lab}.png")

            one_each[lab] = (img, sal)