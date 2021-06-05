# -*- coding: utf-8 -*-

from torch import from_numpy
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import normalize

class My_Compose(Compose):
    '''继承torchvision.transforms.Compose类'''
    def __init__(self, transforms):
        super().__init__(self)
        self.transforms = transforms

    def __call__(self, image):
        '''重写__call__方法'''
        for t in self.transforms:
            image = t(image)
        return image
    
class My_ToTensor(ToTensor):
    '''继承torchvision.transforms.ToTensor类'''
    def __init__(self):
        super().__init__()
        
    def __call__(self, image):
        '''重写__call__方法'''
        return self.to_tensor(image)
    
    @staticmethod
    def to_tensor(pic):
        pic = pic[:, :, None]
        img = from_numpy(pic.transpose((2, 0, 1))).contiguous()
        return img.float().div(4096)

class My_Normalize(Normalize):
    '''继承torchvision.transforms.Normalize类'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        '''重写__call__方法'''
        image = normalize(image, mean=self.mean, std=self.std)
        return image
