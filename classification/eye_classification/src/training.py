from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import helpers

# Training function for regression model
def train_reg_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                num_epochs=25, save_best_param=False, use_gpu=torch.cuda.is_available()):
    """
    Training function for regression approach.
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param dataloaders:
    :param dataset_sizes:
    :param num_epochs:
    :param save_best_param: specifies if parameters should be returned which correspond to the best epoch result or
                            parameters obtained after last training epoch
    :param use_gpu:
    :return model: model with trained parameters
    :return epoch_mae_loss: mean absolute error reached on the validation samples
    :return ground_truth: labels of the validation images
    :return predictions: the networks's predictions for the validation images after the last training epoch
    """

    since = time.time()

    best_model_wts = model.state_dict()
    best_mae_loss = 10.0
    mae = torch.nn.L1Loss(size_average=False)
    ground_truth = []
    predictions = []

    # Check training data distribution
    class_sample_count = [0] * 10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if epoch != 0:
                        scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_mae_loss = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                if phase == 'train':
                    # Count samples per class
                    for item in data[1]:
                        class_sample_count[item] += 1

                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                if isinstance(outputs, tuple):
                    # if outputs contains prediction and sureness-factor, use only predictions for further computations
                    # (to be consistent with the case without sureness-factor)
                    outputs = outputs[0]
                running_mae_loss += mae(outputs, labels.float()).data[0]

                if epoch == (num_epochs - 1) and phase == 'val':
                    ground_truth += list(labels.data.cpu().numpy())
                    predictions += list(outputs.data.cpu().numpy().squeeze())

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_mae_loss = running_mae_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_mae_loss < best_mae_loss:
                best_mae_loss = epoch_mae_loss
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Last MAE: {:.4f}'.format(epoch_mae_loss))
    print('Samples per class in training: {}'.format(class_sample_count))

    # load best model weights
    if save_best_param:
        model.load_state_dict(best_model_wts)
        epoch_mae_loss = best_mae_loss

    return model, epoch_mae_loss, ground_truth, predictions



