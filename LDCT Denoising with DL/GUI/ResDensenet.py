# -*- coding: utf-8 -*-

from torch import cat
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
        return cat([x, new_features], 1)

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

class ResDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4):
        super(ResDenseNet, self).__init__()

        # First convolution
        self.input = nn.Sequential()
        self.input.add_module('conv-input', nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1))
        self.input.add_module('norm-input', nn.BatchNorm2d(num_init_features))
        self.input.add_module('relu-input', nn.ReLU(inplace=True))

        # Each denseblock
        num_layers = block_config[0]
        num_features = num_init_features
        self.DenseBlock1 = _DenseBlock(num_layers, num_features, bn_size, growth_rate)

        num_features = num_features + num_layers * growth_rate
        self.Transiton1 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        num_layers = block_config[1]
        self.DenseBlock2 = block = _DenseBlock(num_layers, num_features, bn_size, growth_rate)

        num_features = num_features + num_layers * growth_rate
        self.Transiton2 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        num_layers = block_config[2]
        self.DenseBlock3 = block = _DenseBlock(num_layers, num_features, bn_size, growth_rate)

        num_features = num_features + num_layers * growth_rate
        self.Transiton3 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        num_layers = block_config[3]
        self.DenseBlock4 = block = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
        num_features = num_features + num_layers * growth_rate
        # Final batch norm
        self.output = nn.Sequential()
        self.output.add_module('conv-output', nn.Conv2d(num_features, 1, 3, padding = 1))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res0 = x
        x = self.input(x)
        res1 = x
        x = self.DenseBlock1(x)
        x = self.Transiton1(x)
        x = self.DenseBlock2(x+res1)
        x = self.Transiton2(x)
        res2 = x
        x = self.DenseBlock3(x)
        x = self.Transiton3(x)
        x = self.DenseBlock4(x+res2)
        features = self.output(x)
        return res0 - features

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
if __name__ == '__main__':
    net = ResDenseNet(growth_rate=16, block_config=(4, 8, 6, 4), num_init_features=64, bn_size=4)
    print(net)