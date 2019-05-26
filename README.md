
# Alternately Updated Convolutional Spectral-Spatial Network for Hyperspectral Image Classification
This is the source code of the paper. We mainly use Tenosorflow to build model and use some functions of Keras.

## Dataset
Before training, you need get all HSI datasets by running download_hsi.py. Or you can download IN, KSC and SS dataset at here [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

## Setup
+ Python 3.5
+ Tensorflow-gpu 1.11.0
+ Keras 2.2.2


To install it and related development package, type:

    pip install numpy scipy matplotlib scikit-learn scikit-image requests
    pip install tensorflow-gpu==1.11.0 keras==2.2.2
   
## Reproducing the results
1) Run the "train.py". You need type the name of HSI dataset. Model are saved in ./models file.

2) Run the "get\_color\_maps.py", for creating the clasification map. You also need type the name of HSI dataset and the time series number of model. And you will get the result in .mat format and classification maps.

## Misc.
Code has been tested under:

+ Windows 10 with 32GB memory, a GTX 1080Ti GPU and Intel i7-8700K CPU.