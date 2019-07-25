
# Alternately Updated Convolutional Spectral-Spatial Network for Hyperspectral Image Classification
This is the source code of the paper. We mainly use Tenosorflow to build model and use some functions of Keras.

Fig 1. Graphic Abstract

## Setup
+ Python 3.5
+ Tensorflow-gpu 1.11.0
+ Keras 2.2.2


To install it and related development package, type:

    pip install numpy scipy matplotlib scikit-learn scikit-image requests
    pip install tensorflow-gpu==1.11.0 keras==2.2.2

You can get more information about installing python and tensorflow-gpu at [here](https://github.com/shuguang-52/FDSSC)
    
## Dataset
Before training, you need get HSI datasets. 

### IN, KSC and SS dataset
You can get IN, KSC and SS dataset by [download_datasets.py](https://github.com/shuguang-52/FDSSC/blob/master/download_datasets.py). Or you can download IN, KSC and SS dataset at here [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).

### Houston dataset
The Houston dataset is provide by the Hyperspectral Image Analysis group and the NSF Funded Center for Airborne Laser Mapping (NCALM) at the University of Houston. The data sets was orginally used for the scientific purposes of the 2013 IEEE GRSS Data Fusion Contest. According to the [terms and conditions](http://hyperspectral.ee.uh.edu/xeadh4f2dftc13/copyright.txt), I cannot distribute the dataset to others. The dataset can be downloaded [here](http://hyperspectral.ee.uh.edu/?page_id=459) subject to the terms and conditions. 

   
## Reproducing the results
1) Run the "train.py". You need type the name of HSI dataset. Model are saved in ./models file.

2) Run the "get\_color\_maps.py", for creating the clasification map. You also need type the name of HSI dataset and the time series number of model. And you will get the result in .mat format and classification maps.

## Classification Result


Fig 2. The classification map of Houston dataset.

## Misc.
Code has been tested under:

+ Windows 10 with 32GB memory, a GTX 1080Ti GPU and Intel i7-8700K CPU.
