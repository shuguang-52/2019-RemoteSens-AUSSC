from osgeo.gdal_array import DatasetReadAsArray
from osgeo import gdal
import scipy.io as sio


houston = gdal.Open("....../2013_DFTC/2013_IEEE_GRSS_DF_Contest_CASI.tif") # Change it
data = DatasetReadAsArray(houston)
print(data.shape, data.dtype)

houston = data.transpose()
print(houston.shape)

sio.savemat('Houston.mat', {'Houston': houston})
