# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pydicom.filereader as dcmreader
import pytorch_ssim
import torch
from scipy import stats
from tqdm import trange

import dataset

class ModelTest():
    '''模型测试，单期模型或双期模型
    Parameters
    ----------
    model_name : str 模型名称
    epoch : int 对应模型的迭代次数
    model_path : str 模型路径
    LDCT_path : str 低剂量图像路径
    NDCT_path : str 正常剂量图像路径
    save_path : str 输出图像保存路径
    stage_num : int 选择模型为(1)单期或(2)双期
    '''
    def __init__(self, model_name, epoch, model_path, LDCT_path, NDCT_path, save_path, stage_num):
        self.model_name = model_name
        self.epoch = epoch
        self.model_path = model_path
        self.LDCT_path = LDCT_path
        self.NDCT_path = NDCT_path
        self.save_path = self.check_path(save_path)
        self.stage_num = stage_num
        self._setup()

    def _setup(self):
        my_totensor  = dataset.My_ToTensor()
        my_normalize = dataset.My_Normalize(0.1225, 0.1188)
        self.transform = dataset.My_Compose([my_totensor])
        self.normalize = dataset.My_Compose([my_normalize])
        self.ssim_loss = pytorch_ssim.SSIM()
        self.color1 = 'royalblue'
        self.color2 = 'teal'
        self.color3 = 'orange'
        self.res_dict = {}
        self.res_dict['model_name'] = self.model_name
        self.res_dict['stage_num'] = self.stage_num
        self.res_dict['epoch'] = self.epoch
        self.res_dict_save_path = r'./{0}/Test_{0}-{1}.json'.format(self.model_name, self.epoch)

    def run(self):
        '''使用此方法开始测试流程'''
        self.load_model()
        self.load_dataset()
        with torch.no_grad():
            self.net_test()

    def net_test(self):
        '''根据stage_num判断单期或双期模型，并开启数据流'''
        if self.stage_num == 1:
            self.onestage_net_test()
        elif self.stage_num == 2:
            self.ts_net_test() 
        else:
            self.stage_num = int(input("Please Enter '1' or '2'(Two-stage model or One-stage model:\n"))
            self.net_test()
    
    def save_res_dict(self, path = None):
        if not path:
            path  = self.res_dict_save_path
        with open(path, 'w') as f:
            json.dump(self.res_dict, f, ensure_ascii = False)

    def onestage_net_test(self):
        '''单期模型数据流'''
        mse_LDCT_list  = []
        mse_Pred_list  = []
        psnr_LDCT_list = []
        psnr_Pred_list = []
        ssim_LDCT_list = []
        ssim_Pred_list = []
        start_time = time()
        for idx in trange(len(self.dataloader)):
            LDCT, NDCT, LD_ds_path = self.dataiter.next()
            LDCT = LDCT.cuda()
            Pred = self.net(LDCT)
            LDCT = LDCT.cpu()
            NDCT = NDCT.cpu()
            Pred = Pred.cpu()
            # 计算SSIM
            ssim_LDCT_list.append(self.SSIM(LDCT, NDCT))
            ssim_Pred_list.append(self.SSIM(Pred, NDCT))
            # 转为array
            LDCT = self.toarray(LDCT)
            NDCT = self.toarray(NDCT)
            Pred = self.toarray(Pred)
            # 计算MSE和PSNR
            LDCT_MSE = self.MSE(LDCT, NDCT)
            Pred_MSE = self.MSE(Pred, NDCT)
            mse_LDCT_list.append(LDCT_MSE)
            mse_Pred_list.append(Pred_MSE)
            psnr_LDCT_list.append(self.PSNR(LDCT, NDCT, mse = LDCT_MSE ))
            psnr_Pred_list.append(self.PSNR(Pred, NDCT, mse = Pred_MSE ))
            # 反标准化
            Pred = self.reverse_normalize(Pred)
            # 保存为Dicom文件
            ds = dcmreader.dcmread(LD_ds_path[0])
            self.save_dcm(ds, idx, Pred, SeriesDescription = 'Predicted_S', SeriesNumber = 3, save_path = self.save_path)
        self.mse_LDCT_np  = np.array(mse_LDCT_list)
        self.mse_Pred_np  = np.array(mse_Pred_list)
        self.psnr_LDCT_np = np.array(psnr_LDCT_list)
        self.psnr_Pred_np = np.array(psnr_Pred_list)
        self.ssim_LDCT_np = np.array(ssim_LDCT_list)
        self.ssim_Pred_np = np.array(ssim_Pred_list)
        self.res_dict['mse_LDCT'] = mse_LDCT_list
        self.res_dict['mse_Pred'] = mse_Pred_list
        self.res_dict['psnr_LDCT'] = psnr_LDCT_list
        self.res_dict['psnr_Pred'] = psnr_Pred_list
        self.res_dict['ssim_LDCT'] = ssim_LDCT_list
        self.res_dict['ssim_Pred'] = ssim_Pred_list

        end_time   = time()
        time_start = datetime.fromtimestamp(start_time)
        time_end   = datetime.fromtimestamp(end_time)
        print('\nTest Start Time:      ' + time_start.strftime('%H:%M:%S.%f'))
        print('Test End Time:        ' + time_end.strftime('%H:%M:%S.%f'))
        print('Test Elapsed Time:    ' + '{:.3f}s'.format(end_time - start_time))
        print('Mean Time(per image): ' + '{:.3f}s'.format((end_time - start_time)/len(self.dataloader)))
        self.res_dict['mean_time'] = r'{:.3f}s'.format((end_time - start_time)/len(self.dataloader))

        print('\nMSE: ')
        print('    LDCT: mean: {} std:{}'.format(self.mse_LDCT_np.mean(), self.mse_LDCT_np.std()))
        print('    Pred: mean: {} std:{}'.format(self.mse_Pred_np.mean(), self.mse_Pred_np.std()))
        print('PSNR:')
        print('    LDCT: mean: {} std:{}'.format(self.psnr_LDCT_np.mean(), self.psnr_LDCT_np.std()))
        print('    Pred: mean: {} std:{}'.format(self.psnr_Pred_np.mean(), self.psnr_Pred_np.std()))
        print('SSIM:')
        print('    LDCT: mean: {} std:{}'.format(self.ssim_LDCT_np.mean(), self.ssim_LDCT_np.std()))
        print('    Pred: mean: {} std:{}'.format(self.ssim_Pred_np.mean(), self.ssim_Pred_np.std()))
        print('\nPaired t-test:')
        print(' MSE(Pred - LDCT): ', stats.ttest_1samp(self.mse_Pred_np - self.mse_LDCT_np, 0))
        print('PSNR(Pred - LDCT): ', stats.ttest_1samp(self.psnr_Pred_np - self.psnr_LDCT_np, 0))
        print('SSIM(Pred - LDCT): ', stats.ttest_1samp(self.ssim_Pred_np - self.ssim_LDCT_np, 0))
        
        plt.figure(figsize = (12,3))
        plt.subplot(131), plt.title('MSE')
        plt.plot(mse_LDCT_list,  color = self.color1, label = 'LDCT')
        plt.plot(mse_Pred_list,  color = self.color3, label = 'Pred')
        plt.legend()
        plt.subplot(132), plt.title('PSNR')
        plt.plot(psnr_LDCT_list, color = self.color1, label = 'LDCT')
        plt.plot(psnr_Pred_list, color = self.color3, label = 'Pred')
        plt.legend()
        plt.subplot(133), plt.title('SSIM')
        plt.plot(ssim_LDCT_list, color = self.color1, label = 'LDCT')
        plt.plot(ssim_Pred_list, color = self.color3, label = 'Pred')
        plt.legend()
        plt.tight_layout(pad=1, w_pad=1.5, h_pad=0.5)
        # plt.savefig('./{0}/Fig_{0}_{1}.jpg'.format(self.model_name, self.epoch), dpi = 500, bbox_inches = 'tight', pad_inches = 0.25)
        plt.show()

    def ts_net_test(self):
        '''双期模型(TS_Net)数据流'''
        mse_LDCT_list  = []
        mse_Pred_F_list  = []
        mse_Pred_S_list  = []
        psnr_LDCT_list = []
        psnr_Pred_F_list = []
        psnr_Pred_S_list = []
        ssim_LDCT_list = []
        ssim_Pred_F_list = []
        ssim_Pred_S_list = []
        start_time = time()
        for idx in trange(len(self.dataloader)):
            LDCT, NDCT, LD_ds_path = self.dataiter.next()
            LDCT = LDCT.cuda()
            Pred_F, Pred_S = self.net(LDCT)
            LDCT = LDCT.cpu()
            NDCT = NDCT.cpu()
            Pred_F = Pred_F.cpu()
            Pred_S = Pred_S.cpu()
            # 计算SSIM
            ssim_LDCT_list.append(self.SSIM(LDCT, NDCT))
            ssim_Pred_F_list.append(self.SSIM(Pred_F, NDCT))
            ssim_Pred_S_list.append(self.SSIM(Pred_S, NDCT))
            # 转为array
            LDCT   = self.toarray(LDCT)
            NDCT   = self.toarray(NDCT)
            Pred_F = self.toarray(Pred_F)
            Pred_S = self.toarray(Pred_S)
            # 计算MSE和PSNR
            LDCT_MSE   = self.MSE(LDCT, NDCT)
            Pred_F_MSE = self.MSE(Pred_F, NDCT)
            Pred_S_MSE = self.MSE(Pred_S, NDCT)
            mse_LDCT_list.append(LDCT_MSE)
            mse_Pred_F_list.append(Pred_F_MSE)
            mse_Pred_S_list.append(Pred_S_MSE)
            psnr_LDCT_list.append(self.PSNR(LDCT, NDCT, mse = LDCT_MSE ))
            psnr_Pred_F_list.append(self.PSNR(Pred_F, NDCT, mse = Pred_F_MSE ))
            psnr_Pred_S_list.append(self.PSNR(Pred_S, NDCT, mse = Pred_S_MSE ))
            # 反标准化
            Pred_S = self.reverse_normalize(Pred_S)
            # 保存为Dicom文件
            ds = dcmreader.dcmread(LD_ds_path[0])
            self.save_dcm(ds, idx, Pred_S, SeriesDescription = 'Predicted_S', SeriesNumber = 3, save_path = self.save_path)
        self.mse_LDCT_np    = np.array(mse_LDCT_list)
        self.mse_Pred_F_np  = np.array(mse_Pred_F_list)
        self.mse_Pred_S_np  = np.array(mse_Pred_S_list)
        self.psnr_LDCT_np   = np.array(psnr_LDCT_list)
        self.psnr_Pred_F_np = np.array(psnr_Pred_F_list)
        self.psnr_Pred_S_np = np.array(psnr_Pred_S_list)
        self.ssim_LDCT_np   = np.array(ssim_LDCT_list)
        self.ssim_Pred_F_np = np.array(ssim_Pred_F_list)
        self.ssim_Pred_S_np = np.array(ssim_Pred_S_list)
        self.res_dict['mse_LDCT'] = mse_LDCT_list
        self.res_dict['mse_Pred_F'] = mse_Pred_F_list
        self.res_dict['mse_Pred_S'] = mse_Pred_S_list
        self.res_dict['psnr_LDCT'] = psnr_LDCT_list
        self.res_dict['psnr_Pred_F'] = psnr_Pred_F_list
        self.res_dict['psnr_Pred_S'] = psnr_Pred_S_list
        self.res_dict['ssim_LDCT'] = ssim_LDCT_list
        self.res_dict['ssim_Pred_F'] = ssim_Pred_F_list
        self.res_dict['ssim_Pred_S'] = ssim_Pred_S_list

        end_time   = time()
        time_start = datetime.fromtimestamp(start_time)
        time_end   = datetime.fromtimestamp(end_time)
        print('\nTest Start Time:      ' + time_start.strftime('%H:%M:%S.%f'))
        print('Test End Time:        ' + time_end.strftime('%H:%M:%S.%f'))
        print('Test Elapsed Time:    ' + '{:.3f}s'.format(end_time - start_time))
        print('Mean Time(per image): ' + '{:.3f}s'.format((end_time - start_time)/len(self.dataloader)))
        self.res_dict['mean_time'] = r'{:.3f}s'.format((end_time - start_time)/len(self.dataloader))
    
        print('\nMSE:')
        print('    LDCT:   mean: {} std:{}'.format(self.mse_LDCT_np.mean(), self.mse_LDCT_np.std()))
        print('    Pred_F: mean: {} std:{}'.format(self.mse_Pred_F_np.mean(), self.mse_Pred_F_np.std()))
        print('    Pred_S: mean: {} std:{}'.format(self.mse_Pred_S_np.mean(), self.mse_Pred_S_np.std()))
        print('PSNR:')
        print('    LDCT:   mean: {} std:{}'.format(self.psnr_LDCT_np.mean(), self.psnr_LDCT_np.std()))
        print('    Pred_F: mean: {} std:{}'.format(self.psnr_Pred_F_np.mean(), self.psnr_Pred_F_np.std()))
        print('    Pred_S: mean: {} std:{}'.format(self.psnr_Pred_S_np.mean(), self.psnr_Pred_S_np.std()))
        print('SSIM:')
        print('    LDCT:   mean: {} std:{}'.format(self.ssim_LDCT_np.mean(), self.ssim_LDCT_np.std()))
        print('    Pred_F: mean: {} std:{}'.format(self.ssim_Pred_F_np.mean(), self.ssim_Pred_F_np.std()))
        print('    Pred_S: mean: {} std:{}'.format(self.ssim_Pred_S_np.mean(), self.ssim_Pred_S_np.std()))
        print('\nPaired t-test:')
        print('MSE(Pred_F - Pred_S):  ', stats.ttest_1samp(self.mse_Pred_F_np - self.mse_Pred_S_np, 0))
        print('PSNR(Pred_F - Pred_s): ', stats.ttest_1samp(self.psnr_Pred_F_np - self.psnr_Pred_S_np, 0))
        print('SSIM(LDCT - Pred_S):   ', stats.ttest_1samp(self.ssim_LDCT_np - self.ssim_Pred_S_np, 0))
        print('SSIM(Pred_F - Pred_S): ', stats.ttest_1samp(self.ssim_Pred_F_np - self.ssim_Pred_S_np, 0))
        
        plt.figure(figsize = (12,3))
        plt.subplot(131), plt.title('MSE')
        plt.plot(mse_LDCT_list,    color = self.color1, label = 'LDCT')
        plt.plot(mse_Pred_F_list,  color = self.color2, label = 'Pred_F')
        plt.plot(mse_Pred_S_list,  color = self.color3, label = 'Pred_S')
        plt.legend()
        plt.subplot(132), plt.title('PSNR')
        plt.plot(psnr_LDCT_list,   color = self.color1, label = 'LDCT')
        plt.plot(psnr_Pred_F_list, color = self.color2, label = 'Pred_F')
        plt.plot(psnr_Pred_S_list, color = self.color3, label = 'Pred_S')
        plt.legend()
        plt.subplot(133), plt.title('SSIM')
        plt.plot(ssim_LDCT_list,   color = self.color1, label = 'LDCT')
        plt.plot(ssim_Pred_F_list, color = self.color2, label = 'Pred_F')
        plt.plot(ssim_Pred_S_list, color = self.color3, label = 'Pred_S')
        plt.legend()
        plt.tight_layout(pad=1, w_pad=1.5, h_pad=0.5)
        plt.savefig('./{0}/Fig_{0}_{1}.jpg'.format(self.model_name, self.epoch), dpi = 500, bbox_inches = 'tight', pad_inches = 0.25)
        plt.show()

    def load_model(self):
        '''加载模型到GPU(可通过继承重写此方法加载自己的模型)'''
        print('Model Name: {}\nEpoch: {}'.format(self.model_name, self.epoch))
        self.net = torch.load(self.model_path)
        self.net.eval()
        self.net.cuda()

    def load_dataset(self):
        '''加载测试图像'''
        test_set = dataset.Mydataset(LDCT_root = self.LDCT_path, NDCT_root = self.NDCT_path, train = False,
                      transform = self.transform, 
                      normalize = self.normalize)
        self.dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1,  num_workers = 2, shuffle = False)
        self.dataiter = iter(self.dataloader)
        print('Image Num: {:5}'.format(test_set.len))
    
    def SSIM(self, input, target):
        '''输入为torch.tensor类型'''
        return self.ssim_loss(input, target).item()

    def PSNR(self, input, target, mse = None):
        '''输入为numpy.array类型'''
        if mse:
            return 10*np.log10(2*2/mse)
        else:
            return 10*np.log10(2*2/self.MSE(input, target))

    @staticmethod
    def check_path(path):
        '''判断路径是否存在，若不存在则创建路径'''
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    @staticmethod
    def MSE(input, target):
        '''输入为numpy.array类型'''
        return np.sum((input-target)**2)/(512*512)

    @staticmethod 
    def Window(WW, WL):
        '''设定窗宽和窗位，返回CT值最大值和最小值的字典'''
        win_dict = {'vmin':WL-WW/2, 'vmax':WL+WW/2}
        return win_dict

    @staticmethod
    def toarray(img_tensor):
        '''torch.tensor To numpy.array'''
        img_tensor = img_tensor.squeeze()
        img_array  = img_tensor.detach().numpy()
        return img_array

    @staticmethod
    def reverse_normalize(img_array):
        '''逆标准化，输出图像矩阵'''
        img_array = (((img_array*0.1188)+0.1225)*4096)
        img_array = np.clip(img_array, 0, 4096)  #防止数据溢出
        img_array = img_array.astype(np.uint16)
        return img_array

    @staticmethod
    def save_dcm(ds, idx, img_array, SeriesDescription, SeriesNumber, save_path):
        '''保存图像为dcm格式'''
        ds.InstitutionName = 'Zhangjinyuan Graduation Design'
        ds.PatientName = 'Test-Patient'
        ds.PatientID = '0000000000'
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.SeriesDescription = SeriesDescription
        ds.SeriesNumber = SeriesNumber
        idx = idx+1
        ds.InstanceNumber = idx
        ds.PixelData = img_array.tobytes()
        ds.Rows, ds.Columns = img_array.shape
        ds.save_as(save_path + '/{}-{}.dcm'.format(SeriesDescription, idx))

if __name__ == '__main__':
    LDCT_path = r'E:\NBIA\LDCT-and-Projection-data\L123\08-23-2018-75696\1.000000-Low Dose Images-07574'
    NDCT_path = r'E:\NBIA\LDCT-and-Projection-data\L123\08-23-2018-75696\1.000000-Full dose images-67226'
    model_name = r'TS_Net_3_r'
    test_epoch = 30
    model_path = r'./{0}/checkpoint/{0}_epoch_{1}.pt'.format(model_name, test_epoch)
    save_path = r'./{}/Test_Result'.format(model_name)
    
    test = ModelTest(model_name, test_epoch, model_path, LDCT_path, NDCT_path, save_path, stage_num=2)
    test.run()
    test.save_res_dict()