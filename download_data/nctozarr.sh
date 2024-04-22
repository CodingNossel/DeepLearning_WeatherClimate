#!/bin/bash

git clone https://github.com/bcdev/nc2zarr.git
cd nc2zarr
pip install numpy pandas pyyaml retry xarray zarr pytest pytest-cov s3fs netcdf4 fsspec dask click
python3 setup.py develop
nc2zarr /data/weather-data/train/*.nc -o /data/weather-data/train.zarr
nc2zarr /data/weather-data/test/*.nc -o /data/weather-data/test.zarr