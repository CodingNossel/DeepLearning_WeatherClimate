import torch
import os

from dataloader.MarsDataset import create_denormalized_matrix_from_tensor
from unet.unet import UNet
from visualizations.visualisation import heat_plotting

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "../model/")
LEVEL_FROM_BOTTOM = 10
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

model = UNet(level=LEVEL_FROM_BOTTOM * 3)
model.load_state_dict(torch.load(CHECKPOINT_PATH + "network_64B_10L_15E_1080T_0.01R_SL1.pt"))
model.eval()


@torch.no_grad()
def forecast(currentState, levels, hours=2):
    """
    Forecasting a weather state.
    :param currentState: current state of the weather
    :param hours: the state of in how many hours shall be predicted; must be dividable by 2
    :return: predicted state of the weather in given hours
    """
    pred = torch.zeros(64, levels*3, 36, 72)
    pred[0] = currentState
    for i in range(int(hours / 2)):
        print("Predicting", i*2+2, "...")
        pred = model(pred)
    pred = create_denormalized_matrix_from_tensor(pred[0].detach(), LEVEL_FROM_BOTTOM)
    heat_plotting(pred, "forecast-temperature", 0, 0)
    heat_plotting(pred, "forecast-wind-x", 1, 0)
    heat_plotting(pred, "forecast-wind-y", 2, 0)
    return pred
