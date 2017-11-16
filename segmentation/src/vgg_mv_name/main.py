# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from training import train_model
from visualize import visualize_model
from net_config import NetConfig

from correspondence_data_loader import CorrespondenceDataLoader

#plt.ion()   # interactive mode


# Settings:
use_gpu = torch.cuda.is_available()
options = NetConfig('config.yaml')

os.makedirs(os.path.split(options.model_path)[0], exist_ok=True)
os.makedirs(options.visualization_dir, exist_ok=True)

_dataloaders = CorrespondenceDataLoader(options).load_data()
dataloaders = dict(zip(['train', 'val'], _dataloaders))
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10, shuffle=True, num_workers=4)
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes
dataset_sizes = dict(zip(['train', 'val'], [len(datal) for datal in _dataloaders]))
with open(os.path.join(options.dataroot[0], "data_dict.json")) as f:
    class_names = json.load(f)

# Network
model = models.vgg16(pretrained=True)

for param in model.features.parameters():   # freeze parameters
    param.requires_grad = False

layers_clf = [
    torch.nn.Linear(25088, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 6)
]
model.classifier = torch.nn.Sequential(*layers_clf)
model.eval()
print(model)

if options.load_model:
    model.load_state_dict(torch.load(options.model_path))
    print('Parameters have been loaded.')

if use_gpu:
    model = model.cuda()

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs

# Training
if not options.load_model:
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, 25, use_gpu)

# Save trained model parameters
if options.save_model:
    torch.save(model.cpu().state_dict(), options.model_path)
    print('Model parameters have been saved.')

# Visualize model predictions and save to file
if options.save_visual:
    visualize_model(model, dataloaders, class_names, os.path.join(options.visualization_dir, "visualization.png"), 6, use_gpu)

# Check accuracy on validation data
running_loss = 0.0
running_corrects = 0
i = 0
#
# for data in dataloaders['val']:
#     # get the inputs
#     inputs, labels = data
#
#     # wrap them in Variable
#     if use_gpu:
#         inputs = Variable(inputs.cuda())
#         labels = Variable(labels.cuda())
#     else:
#         inputs, labels = Variable(inputs), Variable(labels)
#
#     # forward
#     outputs = model(inputs)
#     print(outputs)
#     _, preds = torch.max(outputs.data, 1)
#     loss = criterion(outputs, labels)
#
#     # statistics
#     running_loss += loss.data[0]
#     running_corrects += torch.sum(preds == labels.data)

# total_loss = running_loss / dataset_sizes['val']
# total_acc = running_corrects / dataset_sizes['val']
# print('Results on the {} test images:\n Loss: {:.4f} Acc: {:.4f}'.format(dataset_sizes['val'], total_loss, total_acc))