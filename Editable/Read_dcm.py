import os

from pydicom.filereader import dcmread

class Get_dcm_array():
    ''''''
    def __init__(self, root, openobj):
        ''''''
        self.openobj = openobj
        self.root = root
        self.len = 1
        if openobj == 'folder':
            self.dcm_list = os.listdir(root)
            self.dcm_list = [dcm  for dcm in self.dcm_list if dcm[-4:]=='.dcm']
                          
            self.len = len(self.dcm_list)
            self.dcm_list.sort()

        elif openobj == 'file':
            pass
        if self.len != 0:
            self.get_info()
        
    def getitem(self, index):
        '''根据索引获取image'''
        if self.openobj == 'folder':
            dcm_name = self.dcm_list[index]
            dcm_path = self.root + '\\' + dcm_name
        elif self.openobj == 'file':
            dcm_path = self.root
        return self.get_dcm_array(dcm_path)

    def get_ds_and_array(self, index):
        '''根据索引获取image'''
        if self.openobj == 'folder':
            dcm_name = self.dcm_list[index]
            dcm_path = self.root + '\\' + dcm_name
        elif self.openobj == 'file':
            dcm_path = self.root
        ds = dcmread(dcm_path)
        return ds, ds.pixel_array

    def get_dcm_array(self, path):
        '''读取dcm，并转换像素为CT值'''
        ds = dcmread(path)
        self.ds = ds
        return ds.pixel_array

    def get_info(self):
        if self.openobj == 'folder':
            path = self.root + '\\' + self.dcm_list[0]
        elif self.openobj == 'file':
            path = self.root
        ds = dcmread(path)
        self.WL = 40
        self.WW = 400
        try:
            self.WL = int(ds.WindowCenter)
            self.WW = int(ds.WindowWidth)
        except ValueError:
            self.WL = 40
            self.WW = 400
        except TypeError:
            self.WL = int(ds.WindowCenter[0])
            a = ds.WindowCenter[0]
            self.WW = int(ds.WindowWidth[0])
        except:
            self.WL = 40
            self.WW = 400

if __name__ == '__main__':
    path = r'E:\NBIA\L004\LDCT-and-Projection-data\L004\08-21-2018-84608\1.000000-Low Dose Images'
    dcm_read = Get_dcm_array(path, openobj = 'folder')

