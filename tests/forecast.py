from dataloader.MarsDataset import MarsDataset
from forecasting.forecasting import forecast

dataset = MarsDataset(path_file="../data/my27.zarr", batch_size=1, level_from_bottom=10)

for e in dataset:
    forecast(e[0][0], 10, 6)
    exit()
