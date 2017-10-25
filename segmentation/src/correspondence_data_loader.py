import torch.utils.data
import torchvision.transforms as transforms
from image_folder import CorrespondenceImageFolder
# pip install future --upgrade
from builtins import object
from random import randint


class CorrespondenceDataLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.data_set = None
        self.data_loader = None

        self.initialize(opt)

    def initialize(self, opt):

        # Define Transformations
        if opt.image_width > opt.image_height:
            scale_size = opt.image_height
        else:
            scale_size = opt.image_width

        if opt.input_nc == 3:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif opt.input_nc == 1:
            norm = transforms.Normalize([0], [1])
        else:
            raise(RuntimeError('input_nc not supported'))

        input_transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            norm])

        if opt.output_nc == 3:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif opt.output_nc == 1:
            norm = transforms.Normalize([0], [1])
        else:
            raise(RuntimeError('input_nc not supported'))

        output_transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            norm])

        if opt.phase_test:
            root = opt.test_path
        else:
            root = opt.dataroot + '/A'



        # Create Data Set
        self.data_set = CorrespondenceImageFolder(root=root,
                                                  input_nc=opt.input_nc, output_nc=opt.output_nc,
                                                  input_transform=input_transform, output_transform=output_transform,
                                                  return_masks=not opt.phase_test, return_paths=True,
                                                  train_D=opt.train_D)
        # Create Data Loader
        self.data_loader = torch.utils.data.DataLoader(self.data_set,
                                                       batch_size=opt.batch_size,
                                                       shuffle=opt.shuffle,
                                                       num_workers=0)

    def name(self):
        return 'CorrespondenceDataLoader'

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return len(self.data_set)
