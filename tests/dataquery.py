import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import zarr 
import numpy as np

zarr_data = zarr.open('../data/test.zarr', mode='r')

print(zarr_data.len)

time_data = zarr_data['time'][:]
lon_data = zarr_data['lon'][:]
lat_data = zarr_data['lat'][:]
lev_data = zarr_data['lev'][:]

temp_data = zarr_data['temp'][:]
tsurf_data = zarr_data['tsurf'][:]
co2ice_data = zarr_data['co2ice'][:]
dustcol_data = zarr_data['dustcol'][:]

## Prints
# print(time_data[0])
# print(lon_data[0])
# print(lat_data[0])
# print(lev_data[0])

# print(temp_data[0][0][0][0])
# print(tsurf_data[0][0][0])
# print(co2ice_data[0][0][0])
# print(dustcol_data[0][0][0])

# print(temp_data[time_data[0]][lev_data[0]][lat_data[0]][lon_data[0]])

mars_weather_data = np.empty(shape=((len(time_data) * len(lon_data) * len(lat_data) * len(lev_data)), 5), dtype=np.float32)
        
cur_mat = 0
for time_index, time in enumerate(time_data):
    print(mars_weather_data)
    print(time_index)
    for lev_index, lev in enumerate(lev_data):
        for lat_index, lat in enumerate(lat_data):
            for lon_index, lon in enumerate(lon_data):
                temp_value = temp_data[time_index, lev_index, lat_index, lon_index]

                mars_weather_data[cur_mat, 0] = time
                mars_weather_data[cur_mat, 1] = lev
                mars_weather_data[cur_mat, 2] = lat
                mars_weather_data[cur_mat, 3] = lon
                mars_weather_data[cur_mat, 4] = temp_value
                cur_mat += 1
