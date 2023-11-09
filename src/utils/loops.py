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
    
    if dataset_name == "cifar10" or dataset_name == "mnist":
        classes  = [str(i) for i in range(10)]
    elif dataset_name == "cifar100":
        classes = [str(i) for i in range(100)]


    pos_correct = list(0. for i in range(len(classes)))
    pos_total = list(0. for i in range(len(classes)))
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))


    pos_accuracies = [0]*len(classes)
    acc_per_pos = {}
    class_accuracies = [0]*len(classes)
    acc_per_class = {}
    with torch.no_grad():
        for data in dataloader:
            images, labels, original_labels = data
            
            images = images.to(model.device)
            labels = labels.to(model.device)
            original_labels = original_labels.to(model.device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                pos_correct[label] += c[i].item()
                pos_total[label] += 1

                original_label = original_labels[i]
                class_correct[original_label] += c[i].item()
                class_total[original_label] += 1

    for i in range(len(classes)):

        pos_accuracies[i] = 100 * pos_correct[i] / pos_total[i]
        class_accuracies[i] = 100 * class_correct[i] / class_total[i]

        print('Accuracy of Position %5s : %2d %%' %
            (classes[i], pos_accuracies[i]))
        acc_per_pos[classes[i]] = pos_accuracies[i]
        acc_per_class[classes[i]] = class_accuracies[i]
    
    total_pos_acc = 100 * sum(pos_correct)/sum(pos_total)
    total_class_acc = 100 * sum(class_correct)/sum(class_total)
    print("Total positional accuracy: ", str(total_pos_acc)+"%")
    print("Total class accuracy: ", str(total_class_acc)+"%")


    results = {
        split: {
            "Epoch": epoch,
            "Positional Accuracy": {
                "Total Positional Accuracy": total_pos_acc,
                "Accuracy Per Position": acc_per_pos
            },
            "Class Accuracy": {
                "Total Class Accuracy": total_class_acc,
                "Accuracy Per Class": acc_per_class
            }
        }
    }

    append_json(results, f"experiments/{exp_name}/results.json")

    return results

