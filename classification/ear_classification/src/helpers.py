import torch
from torch.autograd import Variable
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn.modules.module import _addindent
import torch.nn as nn
from torch.nn.modules import loss as L
from skimage.measure import label, regionprops
from PIL import Image


def weights_for_unbalanced_dataset(dataset, nclasses):
    """
    Coumputes weights that can be used to sample elements from underrepresented classes with a higher probability
    :param dataset:
    :param nclasses:
    :return:
    """
    # Count samples per class
    N = len(dataset)
    class_sample_count = [0] * nclasses
    for item in dataset:
        class_sample_count[item[1]] += 1

    weights = 1 / torch.DoubleTensor(class_sample_count)
    weights = weights * N / nclasses
    samples_weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        samples_weight[idx] = weights[val[1]]
    return weights, torch.DoubleTensor(samples_weight)


def smooth_score(class_result):
    """
    Computes sum over probability times index (=score)
    Expected input is probability distribution
    :param class_result:
    :return:
    """
    temp = class_result.clone()
    for i in range(class_result.size()[0]):
        for j in range(class_result.size()[1]):
            temp[i,j] = class_result[i,j].data * j
    return torch.sum(temp,1)


def map_labels(labels):
    """
    Maps predictions and labels to coarse class definitions
    [0,1,2]->0, [3,4,5,6]->1, [7,8,9]->2
    :param labels:
    :return:
    """
    temp = labels.clone()
    for i in range(labels.size()[0]):
        if labels[i] in [0, 1, 2]:
            temp[i] = 0
        elif labels[i] in [3, 4, 5, 6]:
            temp[i] = 1
        elif labels[i] in [7, 8, 9]:
            temp[i] = 2
        else:
            print('No valid input label. Could not map class definitions.')
    return temp


def visualize_results(model, dataloaders, class_names, filename, num_images=6, use_gpu=torch.cuda.is_available()):
    """
    Generates an image which allows to compare a predicted class with the corresponding label for num_images samples
    :param model:
    :param dataloaders:
    :param class_names:
    :param filename:
    :param num_images:
    :param use_gpu:
    :return:
    """
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {} ({:.2f}), label: {}'.format(class_names[preds[j]],
                                                                smooth_score(torch.exp(outputs)).data[j],
                                                                class_names[labels.data[j]]))
            imsave(filename, inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def visualize_results_regression(model, dataloaders, filename, num_images=6, use_gpu=torch.cuda.is_available()):
    """
    Generates an image which allows to compare predicted values with the corresponding labels for num_images samples
    :param model:
    :param dataloaders:
    :param class_names:
    :param filename:
    :param num_images:
    :param use_gpu:
    :return:
    """
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {:.2f}, label: {}'.format(outputs.squeeze().data[j], labels.data[j]))
            imsave(filename, inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def visualize_results_sureness(model, dataloaders, filename, num_images=6, use_gpu=torch.cuda.is_available()):
    """
    Generates an image which allows to compare predicted values with the corresponding labels for num_images samples.
    Additionally, the estimated sureness is given for each image.
    :param model:
    :param dataloaders:
    :param class_names:
    :param filename:
    :param num_images:
    :param use_gpu:
    :return:
    """
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('pred: {:.2f} ({:.2f} sure), label: {}'.format(outputs[0].squeeze().data[j],
                                                                                 outputs[1].squeeze().data[j],
                                                                                 labels.data[j]))
            imsave(filename, inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def imsave(filename, inp, title=None):
    """Imsave for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig(filename)


# Taken from: https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

def print_network(network):
    """
    Static Helper Function to print a network summary
    :param network:
    :return: num_params
    """
    num_params = 0
    for param in network.parameters():
        num_params += param.numel()
    print(network)
    print('Total number of parameters: %d' % num_params)

def save_param(state_dict, param_dir, default):
    """"
    Dialog to save model parameters
    :param state_dict, param_dir, default
    :return None
    """
    yn = input('\nSave current parameters [yes/no]: ')
    if yn == 'yes':
        filename = input('Please enter filename (default={}): '.format(default)) or default
        torch.save(state_dict, param_dir+filename)
        print('Model parameters have been saved.')
    else:
        print('Parameters have NOT been saved.')


def save_summary(net_summary, param_dir, default):
    """"
    Dialog to save model summary
    :param net_summary, param_dir, default
    :return None
    """
    yn = input('\nSave model summary [yes/no]: ')
    if yn == 'yes':
        filename = input('Please enter filename (default={}): '.format(default)) or default
        net_summary.summary2txt(root=param_dir + filename)
        print('Model summary has been saved.')
    else:
        print('Summary has NOT been saved.')


class SurenessMSELoss(nn.Module):
    """
    Combines classic MSELoss function with the sureness factor for regression problems
        loss(x, y)  = 1/n \sum[ (1+s)*|x_i - y_i|^2 - \log((a+s)/b)]
        r(s) := -log((a+s)/b)
        r(0) = m^2 & r(1) = 0 =>
        a = \exp(-m2) / (1 - \exp(-m2))
        b = 1 / (1 - \exp(-m2))
        m2 = alpha^2 * sigma2
        sigma2 = (b - a)^2 / 6
        (this is the prediction error's variance when prediction and label value are assumed to be uniformly
        distributed and stochastically independent (no information of use is contained in the input image))
    """
    def __init__(self, interval, alpha=1, size_average=True):
        super(SurenessMSELoss, self).__init__()
        self.alpha = alpha
        self.sigma2 = (interval[1]-interval[0])**2 / 6
        self.m2 = self.alpha ** 2 * self.sigma2
        self.b = 1 / (1 - math.exp(-self.m2))
        self.a = self.b - 1
        self.size_average = size_average

    def forward(self, input, target):
        pred, sureness = input[0], input[1]
        return self.weightedMSE(pred, target, sureness)

    def weightedMSE(self, pred, target, sureness):
        if self.size_average:
            return sum((1 + sureness) * torch.abs(pred - target) ** 2
                       - torch.log((self.a + sureness) / self.b)) / len(pred)
        else:
            return sum((1 + sureness) * torch.abs(pred - target) ** 2
                       - torch.log((self.a + sureness) / self.b))

def load_state_dict_part(model, pretrained_dict):
    """
    Load only matching parts of a state dict
    :return:
    """
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys (k=key, v=value)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def get_positions(image):
    """
    Computes center positions of image regions
    :param image:
    :return positions:
    """
    label_img = label(image)
    regions = regionprops(label_img)
    positions = []
    for props in regions:
        y, x = props.centroid
        positions.append((x, y))
    return positions

def get_eye_images(image, mask, len):
    images = []
    pos = get_positions(mask)
    for i, coord in enumerate(pos):
        x = coord[0]
        y = coord[1]
        img = image.crop((x-len/2, y-len/2, x+len/2, y+len/2))
        images.append(img)
    return images
