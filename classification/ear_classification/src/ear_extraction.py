# Imports
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from skimage import io, transform
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import skimage as ski
from skimage import color, io, exposure
import numpy as np
from skimage.measure import label, regionprops



def get_ear_images(image, mask, edge=0, resize=None):
    bbox = get_bbox(mask)
    images = []
    for coord in bbox:
        y_min, x_min, y_max, x_max = coord
        if (y_max - y_min) > (x_max - x_min):
            len = y_max - y_min
            y_1 = y_min - edge * len
            y_2 = y_max + edge * len
            x_cent = (x_max + x_min) / 2
            x_1 = x_cent - len * (0.5 + edge)
            x_2 = x_cent + len * (0.5 + edge)
        else:
            len = x_max - x_min
            x_1 = x_min - edge * len
            x_2 = x_max + edge * len
            y_cent = (y_max + y_min) / 2
            y_1 = y_cent - len * (0.5 + edge)
            y_2 = y_cent + len * (0.5 + edge)
        img = image.crop((x_1, y_1, x_2, y_2))
        if not(resize == None):
            img = img.resize((resize, resize))
        images.append(img)
    return images

def save_image(img,filename, cmap):
    plt.imshow(img, cmap=cmap)
    plt.savefig(filename)

def get_bbox(image):
    """
    Returns bounding box coordinates
    :param image:
    :return positions:
    """
    label_img = label(image)
    regions = regionprops(label_img)
    positions = []
    for props in regions:
        y_min, x_min, y_max, x_max = props.bbox
        positions.append((y_min, x_min, y_max, x_max))
    return positions


path = '/home/temp/schneuing/Ohren_Fotos/gesamt/'
filename = 'W0_MGS_MVI_0243_399_4.png'

image = Image.open(path+filename)
mask = io.imread(path+filename.replace('.png', '_mask.png'), as_grey=True)
images = get_ear_images(image, mask, edge=0.25, resize=None)
plt.subplot(1, len(images)+1, 1)
plt.imshow(image)
for i, img in enumerate(images):
    ax = plt.subplot(1, len(images)+1, i+2)
    plt.imshow(img)
plt.savefig('/home/students/schneuing/mouse-project/ear_classification/'+'extracted_ears.png')
