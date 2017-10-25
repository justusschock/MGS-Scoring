import numpy as np
import torch


class BCELoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = torch.nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        # probs        = torch.nn.functional.sigmoid(logits)
        probs_flat   = logits.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


# class DiceLoss(torch.nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, logits, targets):
#         num = targets.size(0)
#         probs = torch.nn.functional.sigmoid(logits)
#         m1  = probs.view(num,-1)
#         m2  = targets.view(num,-1)
#         intersection = (m1 * m2)
#
#         score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
#         score = 1- score.sum()/num
#         return score
#
# class DiceLoss_own(object):
#     def __init__(self, smooth=1e-07):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#
#     @staticmethod
#     def _dice_old(im1, im2, empty_score=1.0):
#         """
#         Computes the Dice coefficient, a measure of set similarity.
#         Parameters
#         ----------
#         im1 : array-like, bool
#             Any array of arbitrary size. If not boolean, will be converted.
#         im2 : array-like, bool
#             Any other array of identical size. If not boolean, will be converted.
#         Returns
#         -------
#         dice : float
#             Dice coefficient as a float on range [0,1].
#             Maximum similarity = 1
#             No similarity = 0
#             Both are empty (sum eq to zero) = empty_score
#
#         Notes
#         -----
#         The order of inputs for `dice` is irrelevant. The result will be
#         identical if `im1` and `im2` are switched.
#         """
#         im1 = np.asarray(im1).astype(np.bool)
#         im2 = np.asarray(im2).astype(np.bool)
#
#         if im1.shape != im2.shape:
#             raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
#
#         im_sum = im1.sum() + im2.sum()
#         if im_sum == 0:
#             return empty_score
#
#         # Compute Dice coefficient
#         intersection = np.logical_and(im1, im2)
#
#         return 1-(2. * intersection.sum() / im_sum)
#
#     @staticmethod
#     def _dice_loss_old(input_tensor, target_tensor):
#         """
#         input_tensor is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
#         target_tensor is a 1-hot representation of the groundtruth, shoud have same size as the input_tensor
#         """
#         assert input_tensor.size() == target_tensor.size(), "Input sizes must be equal."
#         assert input_tensor.dim() == 4, "Input must be a 4D Tensor."
#         uniques = np.unique(target_tensor.data[0])
#         assert set(list(uniques)) <= set([0, 1]), "target_tensor must only contain zeros and ones"
#
#         probs = torch.nn.Softmax(input_tensor)
#         num = probs * target_tensor  # b,c,h,w--p*g
#         num = torch.sum(num, dim=2)
#         num = torch.sum(num, dim=3)  # b,c
#
#         den1 = probs * probs  # --p^2
#         den1 = torch.sum(den1, dim=2)
#         den1 = torch.sum(den1, dim=3)  # b,c,1,1
#
#         den2 = target_tensor * target_tensor  # --g^2
#         den2 = torch.sum(den2, dim=2)
#         den2 = torch.sum(den2, dim=3)  # b,c,1,1
#
#         dice = 2 * (num / (den1 + den2))
#         dice_eso = dice[:, 1]  # we ignore bg dice val, and take the fg
#
#         # dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
#         dice_total = 1 - (torch.sum(dice_eso) / dice_eso.size(0))  # divide by batch_sz
#
#         return dice_total
#
#     def _dice_loss(self, input_tensor, target_tensor):
#         input_f = input_tensor.view(input_tensor.size(0), -1)
#         target_f = target_tensor.view(target_tensor.size(0), -1)
#         prod = torch.mul(input_f, target_f)
#         intersection = torch.sum(prod)
#         dice_coeff = 2 * intersection / (torch.sum(input_f) + torch.sum(target_f) + self.smooth)
#         return dice_coeff
#
#     def __call__(self, gt_tensor, pred_tensor):
#         # if issubclass(type(gt_tensor), torch.CudaFloatTensorBase) or issubclass(type(gt_tensor),
#         #                                                                         torch.FloatTensorBase):
#         #     gt = gt_tensor.data[0]
#         # else:
#         #     gt = gt_tensor
#         #
#         # if issubclass(type(pred_tensor), torch.CudaFloatTensorBase) or issubclass(type(pred_tensor),
#         #                                                                           torch.FloatTensorBase):
#         #     pred = pred_tensor.data[0]
#         # else:
#         #     pred = pred_tensor
#
#         return self._dice_loss(pred_tensor, gt_tensor)

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth_const=1e-4):
        super(DiceLoss, self).__init__()
        self.smooth_const = smooth_const

    def forward(self, input_var, target_var):
        _input_var = input_var.clone()
        _target_var = target_var.clone()
        # _input_var.data = torch.mul(torch.add(_input_var.data, 1), 0.5).float()
        # _target_var.data = torch.mul(torch.add(_target_var.data, 1), 0.5).float()

        result_var = _input_var.clone()
        result_data = torch.mul(_input_var.data, _target_var.data).float()
        result_var.data = result_data

        numerator = result_var.sum().mul(2).add(self.smooth_const)
        denominator = _input_var.sum().add(_target_var.sum().add(self.smooth_const))
        dice_coeff = numerator.div(denominator)

        return 1-dice_coeff


class GANLoss(object):
    """Class to calculate GAN Loss"""
    def __init__(self, loss_fkt=torch.nn.MSELoss, tensor=torch.FloatTensor):
        """
        Function to create and initialize class variables
        :param loss_fkt: function to calculate losses
        :param tensor: Tensor type
        """
        super(GANLoss, self).__init__()
        self.real_label_value = 1.0
        self.fake_label_value = 0.0
        self.Tensor = tensor
        self.loss = loss_fkt()
        self.real_label = None
        self.fake_label = None

    def get_target_tensor(self, input_tensor, target_is_real):
        """
        Function to get the target-tensor (a Tensor of 1s if target is real, a Tensor of 0s otherwise)
        :param input_tensor: the input tensor the target-tensor should be compared with
        :param target_is_real: True if target is real, False otherwise
        :return: target tensor
        """

        target_tensor = None

        if target_is_real:
            create_label = ((self.real_label is None) or
                            (self.real_label.numel() != input_tensor.numel()))

            # No labels created yet or input dim does not match label dim
            if create_label:
                real_tensor = self.Tensor(input_tensor.size()).fill_(self.real_label_value)
                self.real_label = torch.autograd.Variable(real_tensor, requires_grad=False)

            target_tensor = self.real_label

        else:
            # No labels created yet or input dim does not match label dim
            create_label = ((self.fake_label is None) or
                            (self.fake_label.numel() != input_tensor.numel()))

            if create_label:
                fake_tensor = self.Tensor(input_tensor.size()).fill_(self.fake_label_value)
                self.fake_label = torch.autograd.Variable(fake_tensor, requires_grad=False)

            target_tensor = self.fake_label

        return target_tensor

    def __call__(self, input_tensor, target_is_real):
        """
        Function to make class callable
        :param input_tensor: input tensor (result of prediction)
        :param target_is_real: (True if input_tensor is real, False otherwise
        :return: loss value
        """

        target_tensor = self.get_target_tensor(input_tensor, target_is_real)

        # adding sigm when BCELoss is used is not necessary in Original CycleGAN, check why
        # if isinstance(self.loss, torch.nn.BCELoss):
            # sigm = torch.nn.Sigmoid()
            # input_tensor = sigm(input_tensor)
        return self.loss(input_tensor, target_tensor)