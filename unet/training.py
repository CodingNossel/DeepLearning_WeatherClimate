import os
import torch

from dataloader.MarsDataset import MarsDataset
from unet.unet import UNet


def calculate_relative_difference(matrix1, matrix2):
    difference = torch.abs(matrix1 - matrix2)
    relative_difference = difference / matrix2
    return relative_difference


def evaluate(source, calculation, target):
    """
    args:
        source: Source Matrix
        calculation: Calculated Matrix
        target: Target Matrix
    returns:
        difference between calculation and target in perspective to the source
        0: Worse than or equal to source
        0-1: Better than source
        1: Equal to traget
    """
    difference_source_target = calculate_relative_difference(source, target)
    difference_calculation_target = calculate_relative_difference(calculation, target)
    diff = 1 - torch.mean(difference_calculation_target).item() / torch.mean(difference_source_target).item()
    return diff
    # if diff >= 0:
    #    return diff
    # else:
    #    return 0


DATASET_PATH = os.environ.get("PATH_DATASETS", "/data/mars/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "/data/mars/")
EPOCHS = 10
LEARNING_RATE = 0.01
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
LEVEL_FROM_BOTTOM = 5
device = torch.device('cuda' if AVAIL_GPUS else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = MarsDataset(path_file=DATASET_PATH + "train.zarr", batch_size=BATCH_SIZE, level_from_bottom=LEVEL_FROM_BOTTOM)
dataset_test = MarsDataset(path_file=DATASET_PATH + "test.zarr", batch_size=BATCH_SIZE,
                           level_from_bottom=LEVEL_FROM_BOTTOM)

model = UNet(level=LEVEL_FROM_BOTTOM * 3)
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print("Start Epoch {}.".format(epoch + 1))
    model.train()
    epoch_loss = 0
    last_batch = None
    first = True
    for batch in dataset:
        optimizer.zero_grad()
        prediction = model(batch[0])
        label = batch[1]
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= len(dataset)

    model.eval()
    accuracy = 0
    for batch in dataset_test:
        prediction = model(batch[0])
        accuracy += evaluate(batch[0], prediction, batch[1])
    accuracy_percent = accuracy / len(dataset_test)

    print("Epoch {}. Loss: {:.4f}. Accuracy: {:.4f}. Accuracy (%): {:.4f}.".format(epoch + 1, epoch_loss, accuracy,
                                                                                   accuracy_percent))

torch.save(model.state_dict(), CHECKPOINT_PATH + "network.pt")
