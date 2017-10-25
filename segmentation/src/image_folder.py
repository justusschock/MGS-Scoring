################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
from collections import OrderedDict

import torch.utils.data as data

from PIL import Image, ImageOps
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.gif', '.GIF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    returns paths of valid images in dir, sorted
    :param dir:
    :return:
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    images.sort()

    return images


def get_corresponding_path(img_path, label_root_path, train_D=False):
    img_name = (os.path.split(img_path)[-1]).rsplit('.', 1)[0]

    if train_D:
        mask_name = img_name.replace('_pred', '_mask.gif')
    else:
        mask_name = img_name + '_mask.gif'

    return os.path.join(label_root_path, mask_name)


def default_loader(path, n_channels):
    img = Image.open(path)

    if n_channels == 1:
        img = img.convert('L')

    else:
        img = img.convert('RGB')

    return img


def car_loader(path, n_channels):
    img = Image.open(path)

    if n_channels == 1:
        img = img.convert('L')

    else:
        img = img.convert('RGB')

    img = ImageOps.expand(img, border=(1, 0), fill='black')

    return img


class ImageFolder(data.Dataset):

    def __init__(self, root, n_channels, transform=None, return_paths=False,
                 loader=car_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                                                              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.n_channels = n_channels

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path, self.n_channels)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class CorrespondenceImageFolder(data.Dataset):

    def __init__(self, root, input_nc, output_nc, input_transform=None, output_transform=None, return_masks=True, return_paths=False,
                 loader=car_loader, train_D=False):
        img_names = make_dataset(root)
        if len(img_names) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                                                              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.img_names = img_names
        self.return_masks = return_masks
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.return_paths = return_paths
        self.loader = loader
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.train_D = train_D

    def __getitem__(self, index):

        path = self.img_names[index]
        img = self.loader(path, self.input_nc)
        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.return_masks:
            path_mask = get_corresponding_path(path, self.root.replace('/A', '/B'), train_D=self.train_D)
            mask = self.loader(path_mask, self.output_nc)
            if self.output_transform is not None:
                mask = self.output_transform(mask)

            if self.return_paths:
                data_dict = {'img': img, 'mask': mask, 'path_img': path, 'path_mask': path_mask}
            else:
                data_dict = {'img': img, 'mask': mask}

        else:
            if self.return_paths:
                data_dict = {'img': img, 'path_img': path}
            else:
                data_dict = {'img': img}
                
        return data_dict

    def __len__(self):
        return len(self.img_names)
