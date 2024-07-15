<p align="center">
<a href="https://github.com/BennyZhang-Codes/LDCT-denoising-with-Deep-Learning-and-Dicom-Viewer" target="_blank">
	<img src="https://user-images.githubusercontent.com/57568342/120915485-7d27e680-c6d6-11eb-8267-43aa8d59709b.png#pic_center" width=""/>
</a>
</p>

<h1 align="center">LDCT Denoising and Dicom Viewer</h1>
<h3 align="center">Benny&emsp;2021.06.06&emsp;华西坝</h3>
<h3 align="center">West China School of Medcine, Sichuan University.</h3>

## 1 LDCT Denoise
### 1.1 Dataset
Low Dose CT Image and Projection Data([LDCT-and-Projection-data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026)) in The Cancer Imaging Archive(TCIA).
### 1.2 Models
[Trained models](https://github.com/BennyZhang-Codes/LDCT-denoising-with-Deep-Learning-and-Dicom-Viewer/tree/main/LDCT_Denoising/GUI/model)
- TS-Net
- TS-DenseNet
- RED-CNN_MSE
- RED-CNN_SSIM  

![models](https://user-images.githubusercontent.com/57568342/120911784-9d4aac00-c6bc-11eb-9e60-397efa554071.png)

### 1.3 Image Preprocessing
[Image_Preprocessing.ipynb](https://github.com/BennyZhang-Codes/LDCT-denoising-with-Deep-Learning-and-Dicom-Viewer/blob/main/LDCT_Denoising/Image_Preprocessing.ipynb)
![Preprocessing](https://user-images.githubusercontent.com/57568342/120915757-24f1e400-c6d8-11eb-9cc5-c3e7983a037f.png)

### 1.4 Example
![L123_400_40](https://user-images.githubusercontent.com/57568342/120917609-bbc39e00-c6e2-11eb-974e-b01bbc78e0fc.png)

### 1.5 GUI
[Graphical User Interface](https://github.com/BennyZhang-Codes/LDCT-denoising-with-Deep-Learning-and-Dicom-Viewer/tree/main/LDCT_Denoising/GUI)  

![ED](https://user-images.githubusercontent.com/57568342/120898944-cdb02d00-c65f-11eb-9859-38324cc9d418.png)

## 2 Dicom Viewer
### 2.1 Introduction
- View CT images(*.dcm) and save images as other type.
- Dicom Viewer v1.0.exe    [Download](https://github.com/BennyZhang-Codes/LDCT-denoising-with-Deep-Learning-and-Dicom-Viewer/releases/download/v1.0/Dicom.Viewer.1.0.exe)
- [Editable Files](https://github.com/BennyZhang-Codes/LDCT-denoising-with-Deep-Learning-and-Dicom-Viewer/tree/main/Dicom_Viewer/Editable)

### 2.2 Parameters
- Matrix: [64, 2048]
- Window Width: [1, 4096]
- Window Level: [-1024, 3071]
- Save Format: JPG, PNG and BMP. 

![DV](https://user-images.githubusercontent.com/57568342/120813048-1ee8ef80-c580-11eb-9080-c75fbdd60521.png)
