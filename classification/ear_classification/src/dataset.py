from __future__ import print_function, division
import sys
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib
matplotlib.use('Agg')



class MouseEarDataset(Dataset):
    """Mice's ear dataset."""

    def __init__(self, file, root_dir, augmentationFactor=1, transform=None, valTransform=None):
        """
        Args:
            file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            augmentationFactor (int): Factor by which the dataset will be repeated
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.augmentationFactor = augmentationFactor
        self.root_dir = root_dir
        self.inputData = self.__extract_file(file, self.augmentationFactor)
        self.transform = transform
        self.valTransform = valTransform

    def __len__(self):
        return len(self.inputData)

    def __getitem__(self, idx):
        # supports standard indexing
        image = Image.open(self.inputData[idx]['name'])
        bbox = (self.inputData[idx]['y_min'], self.inputData[idx]['x_min'],
                self.inputData[idx]['y_max'], self.inputData[idx]['x_max'])
        image = self.get_ear_image(image, bbox, edge=0.25, resize=None)
        score = self.inputData[idx]['score']

        if self.transform:
            image = self.transform(image)

        sample = (image, score)
        return sample

    def getValItem(self, idx):
        """
        equivalent to __getitem__, but applies optional validation transform instead of transform if given
        :param idx:
        :return:
        """
        image = Image.open(self.inputData[idx]['name'])
        bbox = (self.inputData[idx]['y_min'], self.inputData[idx]['x_min'],
                self.inputData[idx]['y_max'], self.inputData[idx]['x_max'])
        image = self.get_ear_image(image, bbox, edge=0.25, resize=None)
        score = self.inputData[idx]['score']

        if self.valTransform:
            image = self.valTransform(image)
        else:
            if self.transform:
                image = self.transform(image)

        sample = (image, score)
        return sample

    def __extract_file(self, filename, augmentationFactor):
        """
        Reads data from .txt file
        :param filename:
        :param augmentationFactor:
        :return: dictionary containing mice data
        """
        miceData = []
        with open(filename, 'r') as input_data:
            for line in input_data:
                image_name, score = line.split()
                mask_name = image_name.replace('.png', '_mask.png')
                image_name = os.path.join(self.root_dir, image_name)
                mask_name = os.path.join(self.root_dir, mask_name)
                # Consider only scored mice
                if score != '-':
                    score = int(score)
                    y_min, y_max, x_min, x_max = self.get_bbox(mask_name)
                    mouseData = {'name': image_name, 'score': score,
                                 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
                    try:
                        for i in range(augmentationFactor):
                            miceData.append(mouseData)
                    except TypeError:
                        sys.exit('Augmentation factor must be positive integer.')
        return miceData


    def get_bbox(self, mask_path):
        """
        Computes bounding box location
        :param image:
        :return positions:
        """
        mask = io.imread(mask_path, as_grey=True)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return y_min, y_max, x_min, x_max


    def get_ear_image(self, image, bbox, edge=.0, resize=None):
        """
        Creates square-shaped image from given bounding box.
        The length is equal to the maximum bounding box dimension if no additional edges are specified.
        :param image:
        :param bbox:
        :param edge: specifies edges that are added to the cropped image (in percent of original bbox length)
        :param resize:
        :return:
        """
        y_min, x_min, y_max, x_max = bbox
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
        if resize is not None:
            img = img.resize((resize, resize))
        return img
