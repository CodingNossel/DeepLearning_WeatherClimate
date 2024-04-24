import os
import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ConvTranspose2d, ReLU, Module, Sequential
from MarsDataset import MarsDataset


# from visualisation import heat_plotting


class Encoder(Module):
    """
    Encoder module for the U-Net architecture, responsible for downsampling and feature extraction.

    Args:
        inputs (int): Number of input channels/features.
        kernel_size (int or tuple): Size of the convolutional kernels.
    """

    def __init__(self, inputs, kernel_size):
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=inputs, out_channels=inputs, kernel_size=(7, 7)),
            BatchNorm2d(num_features=inputs),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=inputs, out_channels=inputs * 2, kernel_size=(7, 7)),
            BatchNorm2d(num_features=inputs * 2),
            ReLU(inplace=True)
        )
        self.pooling = MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = torch.nn.functional.pad(x, (3, 3, 3, 3), 'circular')
        x = self.conv1(x)
        x = torch.nn.functional.pad(x, (3, 3, 3, 3), 'circular')
        x = self.conv2(x)
        x = self.pooling(x)
        return x


class Decoder(Module):
    """
    Decoder module for the U-Net architecture, responsible for upsampling and feature extraction.

    Args:
        inputs (int): Number of input channels/features.
        kernel_size (int or tuple): Size of the convolutional kernels.
    """

    def __init__(self, inputs, kernel_size):
        super().__init__()
        self.up_conv = ConvTranspose2d(inputs, inputs // 2, kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = Sequential(
            Conv2d(in_channels=inputs // 2, out_channels=inputs // 2, kernel_size=(7, 7)),
            BatchNorm2d(num_features=inputs // 2),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=inputs // 2, out_channels=inputs // 2, kernel_size=(7, 7)),
            BatchNorm2d(num_features=inputs // 2),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_conv(x)
        x = torch.nn.functional.pad(x, (3, 3, 3, 3), 'circular')
        x = self.conv1(x)
        x = torch.nn.functional.pad(x, (3, 3, 3, 3), 'circular')
        x = self.conv2(x)
        return x


class UNet(Module):
    """
    Implementation of a U-Net architecture for semantic segmentation.

    Args:
        kernel_size (int or tuple): Size of the convolutional kernels.
    """

    def __init__(self, kernel_size, level):
        super(UNet, self).__init__()
        self.conv0 = Sequential(
            Conv2d(in_channels=level, out_channels=level, kernel_size=(5, 5)),
            BatchNorm2d(num_features=level),
            ReLU(inplace=True)
        )
        self.enc1 = Encoder(level, kernel_size)
        self.enc2 = Encoder(level * 2, kernel_size)
        self.conv1 = Sequential(
            Conv2d(in_channels=level * 4, out_channels=level * 4, kernel_size=kernel_size),
            BatchNorm2d(num_features=level * 4),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=level * 4, out_channels=level * 4, kernel_size=kernel_size),
            BatchNorm2d(num_features=level * 4),
            ReLU(inplace=True)
        )
        self.dec1 = Decoder(level * 4, kernel_size)
        self.dec2 = Decoder(level * 2, kernel_size)
        self.conv3 = Sequential(
            Conv2d(in_channels=level, out_channels=level, kernel_size=(7, 7)),
            BatchNorm2d(num_features=level),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, (2, 2, 2, 2), 'circular')
        x = self.conv0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'circular')
        x = self.conv1(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'circular')
        x = self.conv2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = torch.nn.functional.pad(x, (3, 3, 3, 3), 'circular')
        x = self.conv3(x)
        return x


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

model = UNet(kernel_size=(3, 3), level=LEVEL_FROM_BOTTOM * 3)
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

# count = 1
# for e in dataset_test:
#    prediction = model(e[0])
#    batch = prediction[0].transpose(0, 1).transpose(1, 2)
#    batch = np.reshape(batch.detach(), (36, 72, 3, 10))
#    heat_plotting(batch, "p" + str(count), 0, 0)
#    count += 1
#    if count == 6:
#        exit()

torch.save(model.state_dict(), CHECKPOINT_PATH + "new_unet.pt")

# loading model:
# model=UNet(input_size=(1,1))
# model.load_state_dict(torch.load(CHECKPOINT_PATH))
# model.eval()
