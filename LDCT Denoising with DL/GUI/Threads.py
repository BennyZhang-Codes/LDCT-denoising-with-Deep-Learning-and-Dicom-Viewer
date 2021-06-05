# -*- coding: utf-8 -*-

import os
from time import time
from datetime import datetime

import numpy as np
from torch import load, no_grad, stack
from PyQt5.QtCore import QThread, pyqtSignal

from dataset import My_Compose, My_Normalize, My_ToTensor

class Denoise():
    def __init__(self, mainwindow):
        self.mainwindow = mainwindow
        self.len = self.mainwindow.dcm_read.len
        self.Model_path = self.mainwindow.model_path
        self.Save_path = self.mainwindow.save_path
        self.Dcm_list = self.mainwindow.dcm_read.dcm_list
        if not os.path.isdir(self.Save_path):
            os.mkdir(self.Save_path)   
        self.net = load(self.Model_path)
        self.net.eval()
        # self.net.cpu()

        my_totensor  = My_ToTensor()
        my_normalize = My_Normalize(0.1225, 0.1188)
        self.transform = My_Compose([my_totensor])
        self.normalize = My_Compose([my_normalize])
    
    def forward(self, idx):
        ds, LDCT_array = self.mainwindow.dcm_read.get_ds_and_array(idx)
        LDCT_array = LDCT_array.astype(np.int16)
        LDCT_array = self.transform(LDCT_array)
        LDCT_array = self.normalize(LDCT_array)
        LDCT_array = stack([LDCT_array])

        with no_grad():
            LDCT_array = LDCT_array.cuda()
            Pred = self.net(LDCT_array)
        Pred = Pred.cpu()
        
        # 转为array
        Pred = self.toarray(Pred)
        
        # 反标准化
        Pred = self.reverse_normalize(Pred)
        
        # 保存为Dicom文件
        self.save_dcm(ds, idx, Pred, SeriesDescription = '{}'.format(self.mainwindow.selected_model_name), SeriesNumber = 3, save_path = self.Save_path)
    
    @staticmethod
    def reverse_normalize(img_array):
        img_array = (((img_array*0.1188)+0.1225)*4096)
        #防止数据溢出
        img_array = np.clip(img_array, 0, 4096)
        img_array = img_array.astype(np.uint16)
        return img_array

    @staticmethod
    def toarray(img_tensor):
        img_tensor = img_tensor.squeeze()
        img_array  = img_tensor.detach().numpy()
        return img_array

    def save_dcm(self, ds, idx, img_array, SeriesDescription, SeriesNumber, save_path):
        ds.InstitutionName = 'Easy-Denoising'

        ds.PatientName = 'Test-Patient'
        ds.PatientID = '0000000000'
        
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.SeriesDescription = SeriesDescription
        ds.SeriesNumber = SeriesNumber
        ds.InstanceNumber = idx+1
        ds.PixelData = img_array.tobytes()
        ds.Rows, ds.Columns = img_array.shape
        ds.save_as(save_path + '/{}'.format(self.Dcm_list[idx]))

class ThreadIconlist(QThread):
    loading_signal = pyqtSignal(int)
    loaded_signal = pyqtSignal()
    def __init__(self, mainwindow):
        super().__init__()
        self.mainwindow = mainwindow
    def run(self):
        self.mainwindow.listWidget.clear()

        for idx in range(self.mainwindow.dcm_read.len):
            self.mainwindow.iconlist(idx)
            self.loading_signal.emit(idx)
        self.mainwindow.actionOpen_Dicom_Folder.setEnabled(True)
        self.mainwindow.actionOpen_Dicom_File.setEnabled(True)
        self.mainwindow.widgetLoading.setHidden(True)

class ThreadDenoise(QThread):
    denoising_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    def __init__(self, mainwindow):
        super().__init__()
        self.mainwindow = mainwindow
    
    def run(self):
        denoise = Denoise(self.mainwindow)
        # print('start')
        start_time = time()
        for idx in range(self.mainwindow.dcm_read.len):
            denoise.forward(idx)
            self.denoising_signal.emit(idx, denoise.Dcm_list[idx])
        end_time   = time()
        time_start = datetime.fromtimestamp(start_time)
        time_end   = datetime.fromtimestamp(end_time)
        

        message = 'Finished!!!\n\n'
        message += 'Start Time:   ' + time_start.strftime('%H:%M:%S.%f') + '\n'
        message += 'End Time:     ' + time_end.strftime('%H:%M:%S.%f') + '\n'
        message += 'Elapsed Time: ' + '{:.3f}s'.format(end_time - start_time) + '\n'
        message += 'Mean Time:    ' + '{:.3f}s'.format((end_time - start_time)/self.mainwindow.dcm_read.len) + '\n'
        self.finished_signal.emit(message)
        # print('Finished!!!')
        # print('Start Time:   ' + time_start.strftime('%H:%M:%S.%f'))
        # print('End Time:     ' + time_end.strftime('%H:%M:%S.%f'))
        # print('Elapsed Time: ' + '{:.3f}s'.format(end_time - start_time))
        # print('Mean Time:    ' + '{:.3f}s'.format((end_time - start_time)/self.mainwindow.dcm_read.len))
        
