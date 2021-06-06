# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import functional as F
import pytorch_ssim

class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction='mean')

class SSIM_Loss(nn.Module):
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, input, target):
        return -self.ssim_loss(input, target)
