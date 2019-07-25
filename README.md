
# Alternately Updated Convolutional Spectral-Spatial Network for Hyperspectral Image Classification
This is the source code of the paper. The Tenosorflow is used to build model.

<div align=center> <img src='classification_maps/Graphical Abstract.png'>
Fig 1. Graphic Abstract
 </div>
## Citation
If you find AUSSC useful in your research, please consider citing.

Chicago/Turabian Style:

Wang, Wenju; Dou, Shuguang; Jiang, Zhongmin; Sun, Liujie.	2018. "A Fast Dense Spectral–Spatial Convolution Network Framework for Hyperspectral Images Classification." Remote Sens. 10, no. 7: 1068.


## Setup
+ Python 3.5+
+ Tensorflow-gpu 1.11.0

To install it and related development package, type:

    pip install numpy scipy matplotlib scikit-learn scikit-image requests
    pip install tensorflow-gpu==1.11.0

You can get more information about installing python and tensorflow-gpu at [here](https://github.com/shuguang-52/FDSSC).
    
## Dataset
Before training, you need get HSI datasets. 

### IN, KSC and SS dataset
You can get IN, KSC and SS dataset by [download_datasets.py](https://github.com/shuguang-52/FDSSC/blob/master/download_datasets.py). Or you can download IN, KSC and SS dataset at here [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).

### Houston dataset
The Houston dataset is provide by the Hyperspectral Image Analysis group and the NSF Funded Center for Airborne Laser Mapping (NCALM) at the University of Houston. The data sets was orginally used for the scientific purposes of the 2013 IEEE GRSS Data Fusion Contest. According to the [terms and conditions](http://hyperspectral.ee.uh.edu/xeadh4f2dftc13/copyright.txt), I cannot distribute the dataset to others. The dataset can be downloaded [here](http://hyperspectral.ee.uh.edu/?page_id=459) subject to the terms and conditions. 

Although I can't provide this data set, I can provide some help when you have got the original data set. The geotif2mat.py can converts the original .tif file to the .mat file. After converting to the .mat file, you can use the train.py to train the Houston dataset. The GDAL(Geospatial Data Abstraction Library) is a translator library for raster geospatial data formats. To run the geotif2mat.py, you need download the GDAL at [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) and type:

     pip install GDAL‑3.0.1‑cp35‑cp35m‑win_amd64.whl

   
## Reproducing the results
1) Run the "train.py". You need type the name of HSI dataset. Model are saved in ./models file.

2) Run the "get\_color\_maps.py", for creating the clasification map. You also need type the name of HSI dataset and the time series number of model. And you will get the result in .mat format and classification maps.

## Classification Result

<img src='classification_maps/hs.png'>
Fig 2. Classification results obtained from the Houston data set using different methods. (a) Ground-truth map. (b) SAE-LR. (c) CNN. (d) SSRN. (e) FDSSC. (f) AUSSC.

## Misc.
Code has been tested under:

+ Windows 10 with 32GB memory, a GTX 1080Ti GPU and Intel i7-8700K CPU.
