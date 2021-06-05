# -*- coding: utf-8 -*-
'''pyuic5 window.ui -o mainwindow.py -x'''

import os
import sys

import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QListWidgetItem, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QMutex, Qt, QSize

from PyQt5.uic import loadUi
from pydicom.filereader import dcmread

from Read_dcm import Get_dcm_array 
from Threads import ThreadIconlist, ThreadDenoise

class Mainwindow(QMainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        loadUi(r'./window_denoise.ui', self)
        self.qmut = QMutex()
        self.ui_setup()
        self.state()
        self.denoised = False

    def ui_setup(self):
        self.lcdNumber.setHidden(True)
        self.verticalScrollBar.setHidden(True)
        # self.tabifyDockWidget(self.dockWidgetDenoise, self.dockWidgetBrowse)
        self.widgetLoading.setHidden(True)
        self.widgetProcessing.setHidden(True)
        self.widgetdenoise.setEnabled(False)

        self.pushButtonStart.setEnabled(False)
        self.horizontalLayoutChooseModel.setEnabled(False)

    def state(self):
        self.WL = 30
        self.WW = 400
        self.HUscale()
        self.matrix = 512
        self.model_path = None
        self.save_path = None
        self.model_choosen = False

    def choosemodel(self, idx):
        TSNet = r'.\model\TS_Net_epoch_30.pt'
        TSDenseNet = r'.\model\TS_DenseNet_epoch_30.pt'
        REDCNNMSE = r'.\model\RED_CNN_MSE_epoch_30.pt'
        REDCNNSSIM = r'.\model\RED_CNN_SSIM_epoch_30.pt'
        model_list = [TSNet, TSDenseNet, REDCNNMSE, REDCNNSSIM, 'Other']
        model_name_list = ['TS-Net', 'TS-DenseNet', 'RED-CNN_MSE', 'RED-CNN_SSIM', 'Other']
        self.model_path = model_list[idx]
        self.selected_model_name = model_name_list[idx]
        if idx == 4:
            self.model_path, self.model_type = QFileDialog.getOpenFileName(self,"Choose a Model",
                                                                     r"C:\Users\Administrator\Desktop\\",
                                                                     "Dicom files(*.pt);;All files(*.*)")
        self.model_choosen = True
        if not self.save_path:
            pass
        elif os.path.exists(self.save_path): 
            self.pushButtonStart.setEnabled(True)
        # print(self.model_path)
    
    def selectsavepath(self):
        self.save_path = QFileDialog.getExistingDirectory(self,"Choose Save Path",r"C:\Users\Administrator\Desktop\\")
        self.lineEditSavepath.setText(self.save_path)
        if self.model_choosen:
            self.pushButtonStart.setEnabled(True)

    def HUscale(self):
        self.CTmin = self.WL - self.WW//2 + 1024
        self.CTmax = self.WL + self.WW//2 + 1024
        if self.CTmin < 0:
            self.CTmin = 0
        if self.CTmax > 4095:
            self.CTmax = 4095

    def IconlistThread(self):
        self.icon_WW = self.WW
        self.icon_CTmin = self.CTmin
        self.icon_CTmax = self.CTmax
        self.widgetLoading.setVisible(True)
        self.progressBarLoading.setMaximum(self.dcm_read.len)
        self.progressBarLoading.setValue(0)
        self.threadiconlist = ThreadIconlist(self)
        self.threadiconlist.loading_signal.connect(self.Loadingprogressbar)
        self.threadiconlist.start()

    def DenoiseThread(self):
        self.pushButtonStart.setEnabled(False)
        self.denoised_dcm_list = []
        self.widgetProcessing.setVisible(True)
        self.actionOpen_Dicom_Folder.setEnabled(False)
        self.actionOpen_Dicom_File.setEnabled(False)
        self.progressBarProcessing.setMaximum(self.dcm_read.len)
        self.progressBarProcessing.setValue(0)
        self.threaddenoise = ThreadDenoise(self)
        self.threaddenoise.denoising_signal.connect(self.Processing)
        self.threaddenoise.finished_signal.connect(self.ProcessFinished)
        self.threaddenoise.start()

    def Loadingprogressbar(self, idx):
        self.progressBarLoading.setValue(idx+1)

    def Processing(self, idx, processing_file):
        self.progressBarProcessing.setValue(idx+1)
        self.labelProcessingfile.setText(processing_file)
        self.denoised_dcm_list.append(processing_file)
        self.denoised = True

        self.verticalScrollBarProcessing.setMaximum(idx+1)
        self.verticalScrollBarProcessing.setValue(idx+1)
        self.verticalScrollBar.setValue(idx+1)
        if idx == 0:
            self.draw_denoised(1)

    def draw_denoised(self, idx):
        try:
            dcm_path = self.save_path + '\\' + self.denoised_dcm_list[idx-1]
        except:
            return
        ds = dcmread(dcm_path)
        img = ds.pixel_array
        img = np.clip(img, self.CTmin, self.CTmax)
        img = img.astype(np.int16)
        img = ((img-self.CTmin)/self.WW)*255

        img = img.astype(np.int8)

        image = QImage(img,img.shape[1], img.shape[0],img.shape[1],
                             QImage.Format_Grayscale8)

        pix   = QPixmap.fromImage(image)
        pix = pix.scaled(self.matrix, self.matrix, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item  = QGraphicsPixmapItem(pix)  #创建像素图元
        scene = QGraphicsScene()          #创建场景
        scene.addItem(item)
        self.graphicsViewProcessing.setScene(scene)      

    def ProcessFinished(self, message_finished):
        self.actionOpen_Dicom_Folder.setEnabled(True)
        self.actionOpen_Dicom_File.setEnabled(True)
        self.widgetProcessing.setHidden(True)
        self.Denoising_Finished(message_finished)
        self.pushButtonStart.setEnabled(True)

    def dcminfo(self):
        infolines = self.dcm_read.ds.formatted_lines()
        info = next(infolines)
        while True:
            try:
                infoline = next(infolines)
                info = info + '\n' + infoline
            except StopIteration:
                break
        self.textBrowser.setText(info)

    def iconlist(self, idx):
        img = self.dcm_read.getitem(idx)
        img = np.clip(img, self.icon_CTmin, self.icon_CTmax)
        img = img.astype(np.int16)
        img = ((img-self.icon_CTmin)/self.icon_WW)*255
        img = img.astype(np.int8)
        image = QImage(img,img.shape[1], img.shape[0],img.shape[1],
                                QImage.Format_Grayscale8)

        pix = QPixmap.fromImage(image)
        pix = pix.scaled(128,128, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        item_icon = QListWidgetItem()
        item_icon.setSizeHint(QSize(150, 150))
        item_icon.setIcon(QIcon(pix))
        item_icon.setText(str(idx+1))
        self.listWidget.addItem(item_icon)
        self.listWidget.update()

    def itemclicked(self):
        item = self.listWidget.selectedItems()[0]
        idx = int(item.text())
        self.lcdNumber.display(idx)
        self.verticalScrollBar.setValue(idx)
        self.draw(idx)
        
    def openfolder(self):
        self.directory = QFileDialog.getExistingDirectory(self,"Choose LDCT Folder",r"C:\Users\Administrator\Desktop\\")
        if os.path.exists(self.directory):     
            self.dcm_read = Get_dcm_array(self.directory, openobj = 'folder')
            try:
                self.enable('folder')
                self.draw(1)
                self.IconlistThread()
            except PermissionError:
                self.openfolder()
        
    def openfile(self):
        self.file_path, self.file_type = QFileDialog.getOpenFileName(self,"Choose a Dicom file",
                                                                     r"C:\Users\Administrator\Desktop\\",
                                                                     "Dicom files(*.dcm);;All files(*.*)")
        if self.file_path and self.file_type == 'Dicom files(*.dcm)':
            self.dcm_read = Get_dcm_array(self.file_path, openobj = 'file')
            self.enable('file')
            self.draw(1)
            self.IconlistThread()
        
    def set_WW(self, ww):
        self.WW = ww
        self.HUscale()
        self.draw(self.idx)
        if self.denoised:
            self.draw_denoised(self.idx)  

    def set_WL(self, wl):
        self.WL = wl
        self.HUscale()
        self.draw(self.idx)
        if self.denoised:
            self.draw_denoised(self.idx)  
    
    def set_Matrix(self, matrix):
        self.matrix = matrix
        self.draw(self.idx)
        if self.denoised:
            self.draw_denoised(self.idx)  
        
    def draw(self, idx):
        self.idx = idx

        img = self.dcm_read.getitem(idx-1)
        img = np.clip(img, self.CTmin, self.CTmax)
        img = img.astype(np.int16)
        img = ((img-self.CTmin)/self.WW)*255

        img = img.astype(np.int8)

        image = QImage(img,img.shape[1], img.shape[0],img.shape[1],
                             QImage.Format_Grayscale8)

        pix   = QPixmap.fromImage(image)
        pix = pix.scaled(self.matrix, self.matrix, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item  = QGraphicsPixmapItem(pix)  #创建像素图元
        scene = QGraphicsScene()          #创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.dcminfo() 

    def enable(self, openobj):
        self.WW = self.dcm_read.WW
        self.WL = self.dcm_read.WL
        self.HUscale()
        
        self.widgetdenoise.setEnabled(True)
        self.graphicsView.setEnabled(True)

        self.horizontalSliderMatrix.setEnabled(True)

        self.tabWidget.setEnabled(True)
        self.listWidget.setEnabled(True)
        self.textBrowser.setEnabled(True)

        self.horizontalSliderWL.setEnabled(True)
        self.horizontalSliderWW.setEnabled(True)
        self.horizontalSliderWW.blockSignals(True)
        self.horizontalSliderWL.blockSignals(True)
        self.horizontalSliderWW.setValue(self.WW)
        self.horizontalSliderWL.setValue(self.WL)
        self.horizontalSliderWW.blockSignals(False)
        self.horizontalSliderWL.blockSignals(False)

        self.labelMatrix.setEnabled(True)
        self.labelmatrix.setEnabled(True)
        self.labelww.setEnabled(True)
        self.labelwl.setEnabled(True)
        self.labelWW.setEnabled(True)
        self.labelWL.setEnabled(True)
        self.labelWW.blockSignals(True)
        self.labelWL.blockSignals(True)
        self.labelWW.setNum(self.WW)
        self.labelWL.setNum(self.WL)
        self.labelWW.blockSignals(False)
        self.labelWL.blockSignals(False)

        self.actionOpen_Dicom_Folder.setEnabled(False)
        self.actionOpen_Dicom_File.setEnabled(False)
     
        if openobj == 'folder':
            self.lcdNumber.setVisible(True)
            self.lcdNumber.display(1)
            self.verticalScrollBar.setVisible(True)
            self.verticalScrollBar.setEnabled(True)
            self.verticalScrollBar.setMaximum(self.dcm_read.len)
            self.verticalScrollBar.setValue(1)

        if openobj == 'file':
            self.lcdNumber.setHidden(True)
            self.verticalScrollBar.setHidden(True)

    def close_confirm(self):
        reply = QMessageBox.question(self, '退出', '确定退出吗？', QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            self.close()

    def about(self):
        QMessageBox.about(self, 'About', '''Easy-Denosing 1.0    --2021-6-5\n
本软件仅适用于dcm格式CT图像的浏览和低剂量CT图像的降噪操作。\n
制作者：张金源（Benny）
邮    箱：823573476@qq.com
地    点：四川大学华西校区男生院男一舍235
All Rights Reserved.''')

    def Denoising_Finished(self, message_finished):
        QMessageBox.information(self, 'Finished', message_finished, QMessageBox.Ok, QMessageBox.Ok)

class RunMainwindow():
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.MainWindow = Mainwindow()
    def run(self):
        self.MainWindow.show()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = Mainwindow()
    MainWindow.showMaximized()
    MainWindow.show()
    sys.exit(app.exec_())
