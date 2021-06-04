# -*- coding: utf-8 -*-
'''pyrcc5 -o images.py images.qrc'''
'''pyuic5 window.ui -o main_window.py -x'''
import os
import sys

import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QListWidgetItem, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.uic import loadUi

from Read_dcm import Get_dcm_array 
from Threads import ThreadIconlist

class Mainwindow(QMainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        loadUi(r'./window.ui', self)
        self.ui_setup()
        self.state()
        

    def ui_setup(self):
        self.widgetsave.setHidden(True)
        self.verticalScrollBar.setHidden(True)
        self.widgetLoading.setHidden(True)

    def state(self):
        self.WL = 30
        self.WW = 400
        self.HUscale()
        self.matrix = 512
        self.save_format = 'jpg'
        self.save_path = '.'


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

    def Loadingprogressbar(self, idx):
        self.progressBarLoading.setValue(idx+1)

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
        self.directory = QFileDialog.getExistingDirectory(self,"Choose Folder",r"C:\Users\Administrator\Desktop\\")
        if os.path.exists(self.directory):  
            self.dcm_read = Get_dcm_array(self.directory, openobj = 'folder')
            if self.dcm_read.len:
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

    def set_WL(self, wl):
        self.WL = wl
        self.HUscale()
        self.draw(self.idx)
    
    def set_Matrix(self, matrix):
        self.matrix = matrix
        self.draw(self.idx)
        
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

    def choose_format(self, idx):
        format_list = ['jpg', 'png', 'bmp']
        self.save_format = format_list[idx]

    def selectsavepath(self):
        self.save_path = QFileDialog.getExistingDirectory(self,"Choose Save Path",r"C:\Users\Administrator\Desktop\\")
        self.lineEditSavepath.setText(self.save_path)

    def set_savepath(self, path):
        if os.path.exists(path):    
            self.save_path = path
            self.pushButtonsave.setEnabled(True)
        elif path == '':
            self.save_path = '.'
            self.pushButtonsave.setEnabled(True)
        else:
            self.pushButtonsave.setEnabled(False)

    def save_image(self):
        img = self.dcm_read.getitem(self.idx-1)
        img = np.clip(img, self.CTmin, self.CTmax)
        img = img.astype(np.int16)
        img = ((img-self.CTmin)/self.WW)*255
        img = img.astype(np.int8)
        image = QImage(img,img.shape[1], img.shape[0],img.shape[1],
                             QImage.Format_Grayscale8)
        image.save(r'{}\{}-WW_{}-WL_{}.{}'.format(self.save_path, self.idx, self.WW, self.WL, self.save_format))

    def enable(self, openobj):
        self.WW = self.dcm_read.WW
        self.WL = self.dcm_read.WL
        self.HUscale()
        
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


        self.labelmatrix.setEnabled(True)
        self.labelww.setEnabled(True)
        self.labelwl.setEnabled(True)
        self.spinBoxMatrix.setEnabled(True)
        self.spinBoxWW.setEnabled(True)
        self.spinBoxWL.setEnabled(True)
        self.spinBoxWW.setValue(self.WW)
        self.spinBoxWL.setValue(self.WL)

        self.actionOpen_Dicom_Folder.setEnabled(False)
        self.actionOpen_Dicom_File.setEnabled(False)
     
        self.widgetsave.setVisible(True)
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
        QMessageBox.about(self, 'About', '''Dicom Viewer 1.0    --2021-5-30\n
本软件仅适用于Dicom文件的浏览和格式转换。\n
制作者：张金源（Benny）
邮    箱：823573476@qq.com
地    点：四川大学华西校区男生院男一舍235
All Rights Reserved.''')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = Mainwindow()
    MainWindow.showMaximized()
    MainWindow.show()
    sys.exit(app.exec_())
