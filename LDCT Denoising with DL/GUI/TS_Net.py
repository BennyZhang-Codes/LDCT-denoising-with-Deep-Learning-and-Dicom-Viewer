# -*- coding: utf-8 -*-

import torch.nn as nn

from ResDensenet import DenseNet, RED_CNN

class TS_DenseNet(nn.Module):
    '''<两期模型(Two-Stage Dense Convolutional Neural Network, TS-DCNN)>
    First Stage : SSIM_Loss\MSE_Loss
    Second Stage : SSIM_Loss\MSE_Loss
    '''
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4):
        super(TS_DenseNet, self).__init__()
        self.F = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        self.S = DenseNet(growth_rate=16, block_config=(3, 4, 4), num_init_features=32, bn_size=4)
    
    def forward(self, x):
        self.F_x = self.F(x)
        x = self.F_x.detach()
        self.S_x = self.S(x)
        return self.F_x, self.S_x

class TS_Net(TS_DenseNet):
    def __init__(self):
        super(TS_Net, self).__init__()

class TS_Net_1(TS_DenseNet):
    def __init__(self):
        super(TS_Net_1, self).__init__()
        self.F = DenseNet(growth_rate=32, block_config=(4, 4, 4, 4), num_init_features=64, bn_size=4)
        self.S = DenseNet(growth_rate=32, block_config=(4, 4, 4, 4), num_init_features=64, bn_size=4)

class TS_Net_2(nn.Module):
    def __init__(self):
        super(TS_Net_2, self).__init__()
        self.F = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        self.S = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        
    def forward(self, x):
        self.F_x = self.F(x)
        x = self.F_x.detach()
        self.S_x = self.S(x)
        return self.S_x

class TS_Net_3(nn.Module):
    def __init__(self):
        super(TS_Net_3, self).__init__()
        self.F = RED_CNN()
        self.S = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        
    def forward(self, x):
        self.F_x = self.F(x)
        x = self.F_x.detach()
        self.S_x = self.S(x)
        return self.S_x