# -*- coding: utf-8 -*-

import os

import numpy as np
import pydicom.filereader

import torch
import torchvision.transforms as transforms

class My_Compose(transforms.Compose):
    '''继承torchvision.transforms.Compose类'''

    def __init__(self, transforms):
        super().__init__(self)
        self.transforms = transforms

    def __call__(self, image):
        '''重写__call__方法'''
        for t in self.transforms:
            image = t(image)
        return image

class My_ToTensor(transforms.ToTensor):
    '''继承torchvision.transforms.ToTensor类'''

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        '''重写__call__方法'''
        return self.to_tensor(image)

    @staticmethod
    def to_tensor(pic):
        pic = pic[:, :, None]
        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        return img.float().div(4096)

class My_Normalize(transforms.Normalize):
    '''继承torchvision.transforms.Normalize类'''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        '''重写__call__方法'''
        image = transforms.functional.normalize(
            image, mean=self.mean, std=self.std)
        return image

class Mydataset(torch.utils.data.Dataset):
    '''读取LDCT和NDCT图像，进行归一化和标准化处理，返回(LDCT, NDCT) if "train = True"或(LDCT, NDCT, LD_ds) if "train = False"元组列表'''
    def __init__(self, LDCT_root, NDCT_root, transform, normalize, train=True):
        '''请指定LDCT和NDCT图像路径，以及图像预处理transform'''
        super().__init__()
        self.LDCT_root = LDCT_root
        self.NDCT_root = NDCT_root
        self.transform = transform
        self.normalize = normalize
        self.train = train

        LDCT_list = os.listdir(LDCT_root)
        NDCT_list = os.listdir(NDCT_root)
        self.data_path = list(zip(LDCT_list, NDCT_list))

        if len(LDCT_list) == len(NDCT_list):
            self.len = len(LDCT_list)
        else:
            print('LDCT和NDCT图像数量不一致，请检查!')

    def __getitem__(self, index):
        '''根据索引获取image和label'''
        LD, ND = self.data_path[index]
        LD_path = self.LDCT_root + '\\' + LD
        ND_path = self.NDCT_root + '\\' + ND
        preprocessed = self.get_preprocess(LD_path, ND_path)
        if self.train:
            return preprocessed[0], preprocessed[1]
        else:
            return preprocessed[0], preprocessed[1], LD_path

    def __len__(self):
        '''返回数据集长度'''
        return self.len

    def get_preprocess(self, LD_path, ND_path):
        '''读取图像并预处理'''
        # 读取
        LD_ds, LD_image = self.get_dcm_array(LD_path)
        ND_ds, ND_image = self.get_dcm_array(ND_path)

        # 归一化和标准化
        LD_image = self.normalize(self.transform(LD_image))
        ND_image = self.normalize(self.transform(ND_image))
        return [LD_image, ND_image, LD_ds]

    @staticmethod
    def get_dcm_array(path):
        '''读取dcm，并转换为像素为CT值'''
        ds = pydicom.filereader.dcmread(path)
        return ds, (ds.pixel_array).astype(np.int16)
