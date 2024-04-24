import torch
import os
from unet.unet import UNet
from visualizations.visualisation import heat_plotting

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "../model/")
LEVEL_FROM_BOTTOM = 5
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

model = UNet(level=LEVEL_FROM_BOTTOM * 3)
model.load_state_dict(torch.load(CHECKPOINT_PATH + "model_unet.pt"))
model.eval()


@torch.no_grad()
def forecast(currentState, hours):
    """
    Forecasting a weather state.
    :param currentState: current state of the weather
    :param hours: the state of in how many hours shall be predicted; must be dividable by 2
    :return: predicted state of the weather in given hours
    """
    pred = currentState
    for i in range(hours / 2):
        pred = model(pred)
    heat_plotting(pred, "forecast")
    return pred
