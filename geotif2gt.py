from osgeo import gdal
from osgeo.gdal_array import DatasetReadAsArray

houston = gdal.Open("Houston_gt.tif")
data = DatasetReadAsArray(houston)

print(data.shape, data.dtype)

houston = data.transpose()
print(houston.shape)

# import matplotlib.pyplot as plt 
# plt.imshow(houston)

houston = houston.reshape(1905*349, 3)

import numpy as np


def list_to_colormap(x_list):
    y = np.zeros(x_list.shape[0])
    for i in range(x_list.shape[0]):
        if (x_list[i] == np.array([0, 0, 0])).all():  #background
            y[i] = 0
        if (x_list[i] == np.array([0, 205, 0])).all(): #grass_healthy
            y[i] = 1
        if (x_list[i] == np.array([127, 255, 0])).all(): #grass_stressed
            y[i] = 2     
        if (x_list[i] == np.array([46, 139, 87])).all(): #grass_synthetic
            y[i] = 3     
        if (x_list[i] == np.array([0, 139, 0])).all(): #tree
            y[i] = 4    
        if (x_list[i] == np.array([160, 82, 45])).all(): #soil
            y[i] = 5    
        if (x_list[i] == np.array([0, 255, 255])).all(): #water
            y[i] = 6    
        if (x_list[i] == np.array([255, 255, 255])).all(): #residential
            y[i] = 7    
        if (x_list[i] == np.array([216, 191, 216])).all(): #commercial
            y[i] = 8    
        if (x_list[i] == np.array([255, 0, 0])).all(): # road
            y[i] = 9    
        if (x_list[i] == np.array([139, 0, 0])).all(): #highway
            y[i] = 10    
        if (x_list[i] == np.array([205, 205, 0])).all(): #railway
            y[i] = 11    
        if (x_list[i] == np.array([255, 255, 0])).all(): #parking_lot1
            y[i] = 12    
        if (x_list[i] == np.array([238, 154, 0])).all(): #parking_lot2
            y[i] = 13    
        if (x_list[i] == np.array([85, 26, 139])).all(): #tennis_court
            y[i] = 14    
        if (x_list[i] == np.array([255, 127, 80])).all(): #running_track
            y[i] = 15    
    return y
    
gt = list_to_colormap(houston)
nb_classes = int(max(gt))
cls, count = np.unique(gt, return_counts=True)
TOTAL_SIZE = np.sum(count[1:])
print(cls, count)
print('The class numbers of the HSI data is:', nb_classes)
print('The total size of the labeled data is:', TOTAL_SIZE)


import scipy.io as sio
gt = gt.reshape(1905, 349)
sio.savemat('datasets/Houston_gt.mat', {'gt': gt})
