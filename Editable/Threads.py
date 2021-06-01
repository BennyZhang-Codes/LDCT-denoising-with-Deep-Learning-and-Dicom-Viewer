import os
from time import time
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal

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
