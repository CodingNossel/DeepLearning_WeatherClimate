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
NORMALIZATION = False

model = UNet(level=LEVEL_FROM_BOTTOM * 3)
model.load_state_dict(torch.load(CHECKPOINT_PATH + "new_unet.pt"))
model.eval()

dataset = MarsDataset(path_file="../data/my27.zarr", batch_size=BATCH_SIZE, level_from_bottom=LEVEL_FROM_BOTTOM)

count = 1
for e in dataset:
    print("Printing ", count, "...")

    if NORMALIZATION:
        batch = e[0][0].transpose(0, 1).transpose(1, 2)
        batch = np.reshape(batch.detach(), (36, 72, 3, LEVEL_FROM_BOTTOM))
    else:
        batch = create_denormalized_matrix_from_tensor(e[0][0], LEVEL_FROM_BOTTOM)

    heat_plotting(batch, "e0-model-1" + str(count), 0, 0)

    if NORMALIZATION:
        batch = e[1][0].transpose(0, 1).transpose(1, 2)
        batch = np.reshape(batch.detach(), (36, 72, 3, LEVEL_FROM_BOTTOM))
    else:
        batch = create_denormalized_matrix_from_tensor(e[1][0], LEVEL_FROM_BOTTOM)

    heat_plotting(batch, "e1-model-1" + str(count), 0, 0)

    prediction = model(e[0])

    if NORMALIZATION:
        batch = prediction[0].transpose(0, 1).transpose(1, 2)
        batch = np.reshape(batch.detach(), (36, 72, 3, LEVEL_FROM_BOTTOM))
    else:
        batch = create_denormalized_matrix_from_tensor(prediction[0].detach(), LEVEL_FROM_BOTTOM)

    heat_plotting(batch, "p-model-1" + str(count), 0, 0)
    count += 1
    if count == 6:
        break
