class NetSummary():
    """
    Class to save all important network and training information
    Provides function to generate an overview as a txt-file
    """
    def __init__(self, network, dataset_root=None, transforms=None, pretrained_param_root=None, epochs=None,
                 batchsize=None, criterion=None, optimizer=None, init_lr=None, scheduler=None, average_mae=None):
        super(NetSummary, self).__init__()
        self.network = network
        self.num_params = self.__get_num_params__()
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.pretrained_param_root = pretrained_param_root
        self.epochs = epochs
        self.batchsize = batchsize
        self.criterion = criterion
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.scheduler = scheduler
        self.average_mae = average_mae

    def summary2txt(self, root):
        file = open(root, 'w')
        file.write('GENERAL NETWORK INFORMATION:\n')
        file.write('Network: {}\n'.format(type(self.network).__name__))
        file.write('{}\n'.format(self.network))
        file.write('Total number of parameters: {}\n'.format(self.num_params))
        file.write('Pretrained parameter set: {}\n'.format(self.pretrained_param_root))

        file.write('\nDATASET:\n')
        file.write('Dataset: {}\n'.format(self.dataset_root))
        file.write('Transformations: {}\n'.format([type(trafo).__name__ for trafo in self.transforms.transforms]))

        file.write('\nTRAINING:\n')
        file.write('Epochs: {}\n'.format(self.epochs))
        file.write('Batchsize: {}\n'.format(self.batchsize))
        file.write('Loss function: {}\n'.format(type(self.criterion).__name__))
        file.write('Optimizer: {}\n'.format(type(self.optimizer).__name__))
        file.write('-> initial learning rate: {}\n'.format(self.init_lr))
        file.write('-> final learning rate: {}\n'.format([group['lr'] for group in self.optimizer.param_groups]))
        file.write('-> momentum: {}\n'.format([group['momentum'] for group in self.optimizer.param_groups]))
        file.write('-> weight decay: {}\n'.format([group['weight_decay'] for group in self.optimizer.param_groups]))
        file.write('Scheduler: {}\n'.format(type(self.scheduler).__name__))
        file.write('Final mean absolute error: {}\n'.format(self.average_mae))
        file.close()

    def __get_num_params__(self):
        num_params = 0
        for param in self.network.parameters():
            num_params += param.numel()
        return num_params
