# Weather Forecasting on Mars

In this project from the course Deep Learning for Weather and Climate at the Otto-von-Guericke-Universität Magdeburg, we are going to use a U-Net to predict the weather. We are Marlon Hörner, Pascal Wissel and Katharina Bade.

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
nc2zarr ./download/*.nc -o marsdata.zarr
```
Note that you should adapt the first path to point to all the .nc files you want to convert.

You can now copy the .zarr file into the /data folder of this repository.

## Usage
When you want to train the UNet you will just need to activate the virtual environment and run the unet.py. Before you can run the unet.py you should check the environment variables in the unet.py file.

## Related Insights
### Weather on Mars
The Weather on Mars is compared to the Weather on Earth more extream. The Mars is generelly cooler than the Earth, because of its lack of an atmosphere and the greater distance to the Sun. When the Sun is shining directly on the Mars at one location, this location gets way hotter and on the Mars the temperature difference is delt with in the same manner as on Earth, but because the temperature difference is so much higher, the winds are also way higher. Sometimes the winds get as high as 400km/h. The mean temperature on Mars is currently around -63°C. Because of its greater distance to the Sun, the Mars has years about twice as long as on Earth and with that seasons about twice as long. 

### Analysis of dataset
TODO    
https://www.sciencedirect.com/science/article/pii/S0032063319303617#sec2    
https://ordo.open.ac.uk/collections/OpenMARS_database/4278950/5 

### Analysis of other forecasting models for Mars
TODO    
https://confluence.ecmwf.int/display/UDOC/MARS+content  
https://ingenology.github.io/mars_weather_api/  
https://www.sciencenews.org/article/perseverance-mars-weather-dust-storms   
https://news.yale.edu/2021/08/30/forecast-mars-otherworldly-weather-predictions 
https://link.springer.com/article/10.1007/s12145-021-00643-0    
https://ieeexplore.ieee.org/document/10002657   
https://ieeexplore.ieee.org/document/10194233   

### Literature
https://www.wetteronline.de/astronews/eisige-staubwueste-ohne-wasser-wetter-und-klima-auf-dem-mars-2018-11-25-ma    
https://www.wetter.de/cms/wetter-auf-dem-mars-eiswolken-gefrorene-gase-und-extreme-temperaturunterschiede-4712745.html  
https://www.futurezone.de/science/article278488/mars-temperatur-klima-und-jahreszeiten-auf-dem-roten-planeten.html  
https://mars.nasa.gov/mars2020/mission/weather/ 

## Analysis of our UNet
TODO

## License
TODO