# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, idx):
        super(_DenseLayer, self).__init__()
        self.add_module('conv{}-1'.format(idx), nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=True))
        self.add_module('norm{}-1'.format(idx), nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu{}-1'.format(idx), nn.ReLU(inplace=True))
        self.add_module('conv{}-2'.format(idx), nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('norm{}-2'.format(idx), nn.BatchNorm2d(growth_rate))
        self.add_module('relu{}-2'.format(idx), nn.ReLU(inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i*growth_rate, growth_rate, bn_size, i+1)
            self.add_module('denselayer-{}'.format(i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('conv-trans', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=True))
        self.add_module('norm-trans', nn.BatchNorm2d(num_output_features))
        self.add_module('relu-trans', nn.ReLU(inplace=True))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential()
        self.features.add_module('conv-input', nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1))
        self.features.add_module('norm-input', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu-input', nn.ReLU(inplace=True))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )
            self.features.add_module('denseblock-{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition-{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('conv1-output', nn.Conv2d(num_features, num_features // 2, 3, padding = 1))
        num_features = num_features // 2
        self.features.add_module('norm-output', nn.BatchNorm2d(num_features))
        self.features.add_module('relu-output', nn.ReLU(inplace=True))
        self.features.add_module('conv2-output', nn.Conv2d(num_features, 1, 3, padding = 1))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        return x - features    

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        # out = self.relu(out)
        return out

class TS_DenseNet(nn.Module):
    def __init__(self):
        super(TS_DenseNet, self).__init__()
        self.F = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        self.S = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        
    def forward(self, x):
        self.F_x = self.F(x)
        x = self.F_x.detach()
        self.S_x = self.S(x)
        return self.F_x, self.S_x

class TS_Net(nn.Module):
    def __init__(self):
        super(TS_Net, self).__init__()
        self.F = RED_CNN()
        self.S = DenseNet(growth_rate=16, block_config=(4, 8, 4), num_init_features=64, bn_size=4)
        
    def forward(self, x):
        self.F_x = self.F(x)
        x = self.F_x.detach()
        self.S_x = self.S(x)
        return self.F_x, self.S_x

if __name__ == '__main__':
    net = DenseNet(growth_rate=16, block_config=(4, 8, 6, 4), num_init_features=64, bn_size=4)
    print(net)