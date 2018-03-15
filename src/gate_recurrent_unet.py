import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_depth, hidden_depth, kernel_size):

        super(ConvGRUCell, self).__init__()

        padding = kernel_size // 2
        self.input_depth = input_depth
        self.hidden_depth = hidden_depth
        self.reset_gate = nn.Conv2d(input_depth + hidden_depth, hidden_depth, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_depth + hidden_depth, hidden_depth, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_depth + hidden_depth, hidden_depth, kernel_size, padding=padding)

        nn.init.orthogonal(self.reset_gate.weight)
        nn.init.orthogonal(self.update_gate.weight)
        nn.init.orthogonal(self.out_gate.weight)
        nn.init.constant(self.reset_gate.bias, 0.)
        nn.init.constant(self.update_gate.bias, 0.)
        nn.init.constant(self.out_gate.bias, 0.)

    def forward(self, input_tensor, prev_state=None):

        batch_size = input_tensor.data.size()[0]
        spatial_size = input_tensor.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_depth] + list(spatial_size)

            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        stacked_inputs = torch.cat([input_tensor, prev_state], dim=1)

        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_tensor, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRUCellTranspose(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_depth, hidden_depth, kernel_size):

        super(ConvGRUCellTranspose, self).__init__()

        padding = kernel_size // 2
        self.input_depth = input_depth
        self.hidden_depth = hidden_depth
        self.reset_gate = nn.Conv2dTranspose(input_depth + hidden_depth, hidden_depth, kernel_size, padding=padding)
        self.update_gate = nn.Conv2dTranspose(input_depth + hidden_depth, hidden_depth, kernel_size, padding=padding)
        self.out_gate = nn.Conv2dTranspose(input_depth + hidden_depth, hidden_depth, kernel_size, padding=padding)

        nn.init.orthogonal(self.reset_gate.weight)
        nn.init.orthogonal(self.update_gate.weight)
        nn.init.orthogonal(self.out_gate.weight)
        nn.init.constant(self.reset_gate.bias, 0.)
        nn.init.constant(self.update_gate.bias, 0.)
        nn.init.constant(self.out_gate.bias, 0.)

    def forward(self, input_tensor, prev_state=None):

        batch_size = input_tensor.data.size()[0]
        spatial_size = input_tensor.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_depth] + list(spatial_size)

            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        stacked_inputs = torch.cat([input_tensor, prev_state], dim=1)

        update =  F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_depth, hidden_depths, kernel_sizes, n_layers, transpose=False):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_depth = input_depth

        if type(hidden_depths) != list:
            self.hidden_depths = [hidden_depths]*n_layers
        else:
            assert len(hidden_depths) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_depths = hidden_depths
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_depth
            else:
                input_dim = self.hidden_depths[i-1]
            if not transpose:
                cell = ConvGRUCell(input_dim, self.hidden_depths[i], self.kernel_sizes[i])
                name = 'ConvGRUCell_' + str(i).zfill(2)
            else:
                cell = ConvGRUCellTranspose(input_dim, self.hidden_depths[i], self.kernel_sizes[i])
                name = 'ConvGRUCellTranspose_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, input_tensor, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if hidden is None:
            hidden = [None]*self.n_layers

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_tensor, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_tensor = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


class GateRecurrentDownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(GateRecurrentDownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv_gru = ConvGRU(in_channels, out_channels, kernel_sizes=3, n_layers=2)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_tensor, prev_state=None):
        x = self.conv_gru(input_tensor, prev_state)[-1]
        before_pool = x

        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        def conv3x3(in_channels, out_channels, stride=1,
                    padding=1, bias=True, groups=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups)

        def upconv2x2(in_channels, out_channels, mode='transpose'):
            if mode == 'transpose':
                return nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2)
            else:
                # out_channels is always going to be the same
                # as in_channels
                return nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    conv1x1(in_channels, out_channels))

        def conv1x1(in_channels, out_channels, groups=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
                stride=1)

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class GateRecurrentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat', gpu_ids=[]):
        super(GateRecurrentUNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.sigm = torch.nn.Sigmoid()
        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = GateRecurrentDownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = nn.Conv2d(
            outs,
            out_channels,
            kernel_size=1,
            groups=1,
            stride=1)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

        self.cuda()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input_tensor, prev_states=None):
        if not isinstance(prev_states, list):
            prev_states = [prev_states]*(len(self.down_convs))

        assert len(prev_states) == len(self.down_convs), "Wrong number of previous states"

        encoder_outs = []
        next_states = []

        x = input_tensor

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x, prev_states.pop(0))
            next_states.append(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return self.sigm(x), next_states


def test(num_seqs, channels_img_in, channels_img_out, size_image, model_depth, n_filters_start, max_epoch, cuda_test):
    model = GateRecurrentUNet(in_channels=channels_img_in, out_channels=channels_img_out, depth=model_depth,
                              start_filts=n_filters_start, up_mode='transpose',
                              merge_mode='concat', gpu_ids=[])
    input_image = torch.rand(num_seqs, 1, channels_img_in, size_image, size_image)
    target_image = torch.rand(num_seqs, 1, channels_img_out, size_image, size_image)

    print('\n\n ==> Create Autograd Variables:')
    input_gru = Variable(input_image)
    target_gru = Variable(target_image)
    if cuda_test:
        input_gru = input_gru.cuda()
        target_gru = target_gru.cuda()

    print('\n\n ==> Create a MSE criterion:')
    mse_criterion = nn.MSELoss()
    err = 0

    for e in range(max_epoch):
        for time in range(num_seqs):
            h_next = model(input_gru[time], None)
            err += mse_criterion(h_next[0], target_gru[time])
            print(err.data[0])

test(num_seqs=100, channels_img_in=3, channels_img_out=1, size_image=100, model_depth=3, n_filters_start=64, max_epoch=10, cuda_test=True)
