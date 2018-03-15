import os
from torchvision import transforms
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bland_altman import bland_altman_plot

import dataset
import helpers
import custom_transforms
import training
import regNet
import net_summary



# Settings
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
use_gpu = False
root_dir = '../../../eye_images/'
param_dir = '../../'
use_sureness = True  # Defines if modified network structure and loss function with sureness-factor are used
epochs = 35
batch_size = 32
lr = 0.001
k = 30
test_mode = False  # Set True if only first iteration of k-fold validation should be executed




# Dataloading
data_transforms = transforms.Compose([
    custom_transforms.RandomRotation([-30, 30]),
    transforms.CenterCrop(60),
    transforms.RandomCrop(50),
    custom_transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# No random transformations applied to the validation images
val_transforms = transforms.Compose([
    transforms.CenterCrop(50),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eye_dataset = dataset.MouseEyeDataset(file=root_dir+'MGS.txt', root_dir=root_dir, augmentationFactor=1,
                                      transform=data_transforms, valTransform=val_transforms)

# Network
pretr_param_dir = param_dir + 'params_tutorialNet_CIFAR10.pth'

if use_sureness:
    model = regNet.Net_Sure()
else:
    model = regNet.Net()

model = helpers.load_state_dict_part(model, torch.load(pretr_param_dir))

# Freeze parameters
for param in model.features.parameters():
    param.requires_grad = False

# Print model summary
helpers.print_network(model)

init_model_wts = model.state_dict()


labels = []
preds = []

best_model_wts = init_model_wts

# K-Fold validation
kf = KFold(n_splits=k, shuffle=True)

for run, split in enumerate(kf.split(eye_dataset)):
    train_idx, val_idx = split
    print('\nRun {}/{}'.format(run, kf.get_n_splits(eye_dataset) - 1))

    train_set, val_set = [eye_dataset[i] for i in train_idx], [eye_dataset.getValItem(j) for j in val_idx]
    image_datasets = {'train': train_set, 'val': val_set}
    _, samples_weight = helpers.weights_for_unbalanced_dataset(image_datasets['train'], 10)
    # Sampler is used to compensate for unbalanced dataset
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False,
                                       sampler=sampler, num_workers=4),
                   'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4)}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model.load_state_dict(init_model_wts)

    if use_gpu:
        model = model.cuda()

    # Criterion
    if use_sureness:
        criterion = helpers.SurenessMSELoss(interval=[0, 9], alpha=0.9)
    else:
        criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
    # Training
    model, mae_loss, ground_truth, prediction = training.train_reg_model(model, criterion, optimizer, scheduler,
                                                                         dataloaders, dataset_sizes, epochs, False,
                                                                         use_gpu)

    labels += ground_truth
    preds += prediction

# Compute Mean Absolute Error over all validation samples
overall_mae = mean_absolute_error(np.asarray(labels), np.asarray(preds))
print('overall mae: {:.4f}'.format(overall_mae))
# Save trained model parameters
model = model.cpu()
helpers.save_param(model.state_dict(), param_dir,
                   type(model).__name__ + '[' + str(round(overall_mae, 4)) + 'mae]_params.pth')
# Save training summary
results = net_summary.NetSummary(network=model,
                                 dataset_root=root_dir,
                                 transforms=data_transforms,
                                 pretrained_param_root=pretr_param_dir,
                                 epochs=epochs,
                                 batchsize=batch_size,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 init_lr=lr,
                                 scheduler=scheduler,
                                 average_mae=overall_mae)
helpers.save_summary(results, param_dir,
                     type(model).__name__ + '[' + str(round(overall_mae, 4)) + 'mae]_summary.txt')
# Bland Altman plot
print(labels)
print(len(labels))
print(preds)
print(len(preds))
bap = bland_altman_plot(np.asarray(preds), np.asarray(labels))
plt.xlim([0, 9])
plt.savefig(os.path.split(os.getcwd())[0] + '/bland-altman-plot.png')


print("")
