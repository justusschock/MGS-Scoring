import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
import os
import regNet
import helpers
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def validate(model, val_loader, criterion):
    """
    Returns validation loss and prints current accuracy on the testset
    Args:
        model
        val_loader
        criterion
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    for data in val_loader:
        images, labels = data
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        running_loss += loss.data[0]
    val_loss = running_loss / total
    print('Accuracy of the network on the {} test images: {:.2f} %'.format(total, 100 * correct / total))
    return val_loss


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
# Settings
epochs = 50
batchsize = 32
param_dir = '/home/temp/schneuing/Ohren_Ergebnisse/'
path = '/home/temp/schneuing/cifar10/'
stats_per_epoch = 5


trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Network
net = regNet.Net()

net.classifier = nn.Sequential(
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

net.cuda()

helpers.print_network(net)

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

# Training
for epoch in range(epochs):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    net.train(True)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        var = int(len(trainset) / batchsize / stats_per_epoch)
        if i % var == (var-1):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / var))
            running_loss = 0.0
    val_loss = validate(net, testloader, criterion)
    scheduler.step(val_loss)

print('Finished Training')


# Performance check
validate(net, testloader, criterion)

# Classwise performance
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# Test results
dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(Variable(images.cuda()))
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Save model parameters
net = net.cpu()
helpers.save_param(net.state_dict(), param_dir, 'params_Net_CIFAR10.pth')

print("")
