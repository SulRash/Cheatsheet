import torch
import torch.nn as nn

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
    return loss

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

def test(model, dataloader, epoch, exp_name, dataset_name: str, split: str = "test"):
    
    if dataset_name == "cifar10":
        classes  = [str(i) for i in range(10)]
    elif dataset_name == "pets":
        classes = [str(i) for i in range(37)]
    elif dataset_name == "cifar100":
        classes = [str(i) for i in range(100)]

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    accuracies = [0]*len(classes)
    acc_per_class = {}
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
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
        split: {
            "Epoch": epoch,
            "Total Accuracy": total_acc,
            "Accuracy Per Class": acc_per_class
        }
    }

    append_json(results, f"experiments/{exp_name}/results.json")

    return total_acc

