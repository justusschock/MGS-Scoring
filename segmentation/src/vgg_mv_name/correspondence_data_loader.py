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
            transforms.Scale((scale_size, scale_size)),
            transforms.ToTensor(),
            norm])

        output_transform = torch.from_numpy

        if opt.phase_test:
            root = [opt.test_path]
        else:

            root = opt.dataroot

        self.data_set, self.data_loader = [], []

        for path in root:

        # Create Data Set
            dataset = CorrespondenceImageFolder(root=path,
                                                input_nc=opt.input_nc,
                                                input_transform=input_transform, output_transform=output_transform,
                                                return_paths=True)
            self.data_set.append(dataset)
            # Create Data Loader
            self.data_loader.append(torch.utils.data.DataLoader(dataset,
                                                           batch_size=opt.batch_size,
                                                           shuffle=opt.shuffle,
                                                           num_workers=0))

    def name(self):
        return 'CorrespondenceDataLoader'

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return [len(dataset) for dataset in self.data_set]