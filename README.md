# Weather Forecasting on Mars

Marlon Hörner, Pascal Wissel, Katharina Bade\
Otto-von-Guericke-Universität Magdeburg\
April 2024\
Deep Learning for Weather and Climate

## Installation
We recomend that you use a virtual environment with Python 3.10. After you have setup the virtual environment, you can install the required packages with
```bash
pip install -r requirements.txt
```
You will need a Dataset to run the UNet. You can download it from [here](https://ordo.open.ac.uk/collections/OpenMARS_database/4278950/5). The Data is in the .nc file format. For better and faster access, we are using the .zarr file format. To convert the .nc file to .zarr file, you can use this [repository](https://github.com/bcdev/nc2zarr.git). When you have downloaded the repository you can install the required packages.
```bash
pip install numpy pandas pyyaml retry xarray zarr pytest pytest-cov s3fs netcdf4 fsspec dask click
```
For the repository to run you will need to go to the root for the repository and run
```bash
python setup.py develop
```
To run the conversion of the .nc file to .zarr file, you can use the following command
```bash
nc2zarr ./download/*.nc -o marstraining.zarr
```
You can now copy the .zarr file into the /data folder of this repository.

## Usage
When you want to train the UNet you will just need to activate the virtual environment and run the unet.py. Before you can run the unet.py you should check the environment variables in the unet.py file.

### Weather on Mars

https://www.wetteronline.de/astronews/eisige-staubwueste-ohne-wasser-wetter-und-klima-auf-dem-mars-2018-11-25-ma
https://www.wetter.de/cms/wetter-auf-dem-mars-eiswolken-gefrorene-gase-und-extreme-temperaturunterschiede-4712745.html
https://www.futurezone.de/science/article278488/mars-temperatur-klima-und-jahreszeiten-auf-dem-roten-planeten.html
https://mars.nasa.gov/mars2020/mission/weather/

### Analysis of dataset

https://www.sciencedirect.com/science/article/pii/S0032063319303617#sec2
https://ordo.open.ac.uk/collections/OpenMARS_database/4278950/5

### Analysis of other forecasting models for Mars

https://confluence.ecmwf.int/display/UDOC/MARS+content
https://ingenology.github.io/mars_weather_api/
https://www.sciencenews.org/article/perseverance-mars-weather-dust-storms
https://news.yale.edu/2021/08/30/forecast-mars-otherworldly-weather-predictions
https://link.springer.com/article/10.1007/s12145-021-00643-0
https://ieeexplore.ieee.org/document/10002657
https://ieeexplore.ieee.org/document/10194233

### Analysis of our GNN

### Notes

The Healpy Lib only works on Linux and Mac
