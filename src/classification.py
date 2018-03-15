import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from skimage.measure import label, regionprops
from skimage import io
import os

import regNet


def load_network(model_path):

    model = regNet.Net()

    model.load_state_dict(torch.load(os.path.join(model_path, 'regNet.pth')))
    model.eval()
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
    return images, pos

