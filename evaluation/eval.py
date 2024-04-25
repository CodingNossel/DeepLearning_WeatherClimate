import torch
import os
import numpy as np

from dataloader.MarsDataset import MarsDataset, create_denormalized_matrix_from_tensor
from unet.unet import UNet
from visualizations.visualisation import heat_plotting

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "../model/")
LEVEL_FROM_BOTTOM = 5
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

model = UNet(level=LEVEL_FROM_BOTTOM * 3)
model.load_state_dict(torch.load(CHECKPOINT_PATH + "new_unet.pt"))
model.eval()

dataset = MarsDataset(path_file="../data/my27.zarr", batch_size=BATCH_SIZE, level_from_bottom=LEVEL_FROM_BOTTOM)

count = 1
for e in dataset:
    print("Printing ", count, "...")

    batch = e[0][0].transpose(0, 1).transpose(1, 2)
    batch = np.reshape(batch.detach(), (36, 72, 3, LEVEL_FROM_BOTTOM))
    heat_plotting(create_denormalized_matrix_from_tensor(batch, LEVEL_FROM_BOTTOM), "e0-model-1" + str(count), 0, 0)

    batch = e[1][0].transpose(0, 1).transpose(1, 2)
    batch = np.reshape(batch.detach(), (36, 72, 3, LEVEL_FROM_BOTTOM))
    heat_plotting(create_denormalized_matrix_from_tensor(batch, LEVEL_FROM_BOTTOM), "e1-model-1" + str(count), 0, 0)

    prediction = model(e[0])
    batch = prediction[0].transpose(0, 1).transpose(1, 2)
    batch = np.reshape(batch.detach(), (36, 72, 3, LEVEL_FROM_BOTTOM))
    heat_plotting(create_denormalized_matrix_from_tensor(batch, LEVEL_FROM_BOTTOM), "p-model-1" + str(count), 0, 0)
    count += 1
    if count == 6:
        break
