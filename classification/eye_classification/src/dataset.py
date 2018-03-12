from __future__ import print_function, division
import sys
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib
matplotlib.use('Agg')




class MouseEyeDataset(Dataset):
    """Mice's eye dataset."""

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
        self.inputData = self.__extract_file(file, self.augmentationFactor)
        self.root_dir = root_dir
        self.transform = transform
        self.valTransform = valTransform

    def __len__(self):
        return len(self.inputData)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.inputData[idx]['name'])
        image = Image.open(img_name)
        image = self.__get_eye_image(image, self.inputData[idx]['x_pos'], self.inputData[idx]['y_pos'], 120)
        score = self.inputData[idx]['score']

        if self.transform:
            image = self.transform(image)

        sample = (image, score)
        return sample

    def getValItem(self, idx):
        """
        equivalent to __getitem__, but applies optional validation transform instead of transform (if given)
        """
        img_name = os.path.join(self.root_dir, self.inputData[idx]['name'])
        image = Image.open(img_name)
        image = self.__get_eye_image(image, self.inputData[idx]['x_pos'], self.inputData[idx]['y_pos'], 120)
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
                name, mouse_no, score, position = line.split()
                mouse_no = int(mouse_no)
                # Consider only scored mice
                if score != '-':
                    score = int(score)
                    x, y = position.split(',')
                    x = int(x.replace('(', ''))
                    y = int(y.replace(')', ''))
                    mouseData = {'name': name, 'mouse_no': mouse_no, 'score': score, 'x_pos': x, 'y_pos': y}
                    try:
                        for i in range(augmentationFactor):
                            miceData.append(mouseData)
                    except TypeError:
                        sys.exit('Augmentation factor must be positive integer.')
        return miceData

    def __get_eye_image(self, img, x, y, len):
        return img.crop((y-len/2, x-len/2, y+len/2, x+len/2))