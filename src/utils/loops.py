import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck')

def train_cifar(model, train_dataloader):
    for step, batch in enumerate(train_dataloader):
        inputs = batch[0].to(model.device)
        labels = batch[1].to(model.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        model.backward(loss)
        model.step()

def valid_cifar(model, valid_dataloader):
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

def test_cifar(model, test_dataloader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
            (classes[i], 100 * class_correct[i] / class_total[i]))