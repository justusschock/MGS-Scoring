import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from net_config import NetConfig
from segmentation_nn import SegmentationNetwork
from classification import load_network, get_eye_images
from image_handler import ImageHandler
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import numpy as np
import time


class Backend(object):
    def __init__(self):
        self.options = NetConfig('seg_config.yaml')
        self.gpu_ids = [0]

        self.seg_model = SegmentationNetwork(options=self.options, gpu_ids=self.gpu_ids)

        self.reg_transforms = transforms.Compose([
            transforms.CenterCrop(50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.reg_model = load_network(self.options.model_path)

    def predict(self, imgs):
        if isinstance(imgs, list):
            img_list = imgs
        elif isinstance(imgs, np.ndarray) and len(imgs.shape) == 4:
            img_list = list(imgs)
        else:
            img_list = [imgs]

        imgs_masks = self.seg_model.predict(img_list)

        for idx, data in enumerate(imgs_masks):
            image, mask = data
            orig_size = img_list[idx].size

            mask = ((ImageHandler._tensor_to_image(mask[0], mask=True) > 127)*255).astype(np.uint8)
            if mask.any():

                images, positions = get_eye_images(image, mask, 120)

                orig_pos = []

                curr_img_size = image.size

                for pos_idx in range(len(positions)):
                    _tmp = []
                    for coord_idx in range(len(orig_size)):
                        # calculate original position for each coordinate
                        _tmp.append(positions[pos_idx][coord_idx]*orig_size[coord_idx]/curr_img_size[coord_idx])
                    orig_pos.append(tuple(_tmp))

                for i, img in enumerate(images):
                    images[i] = self.reg_transforms(img)

                images = torch.stack(images, 0)

                preds = self.reg_model(Variable(images))
                positions = orig_pos
                result = float(torch.mean(preds.data))
            else:
                result = None
                positions = []
            return {'value': result, 'coords': positions}
