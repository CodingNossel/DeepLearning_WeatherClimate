import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ConvTranspose2d, ReLU, Module, Sequential


class Encoder(Module):
    """
    Encoder module for the U-Net architecture, responsible for downsampling and feature extraction.

    Args:
        inputs (int): Number of input channels/features.
    """

    def __init__(self, inputs):
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
    """

    def __init__(self, inputs):
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
    Implementation of a network architecture for semantic segmentation.
    """

    def __init__(self, level):
        super(UNet, self).__init__()
        self.conv0 = Sequential(
            Conv2d(in_channels=level, out_channels=level, kernel_size=(5, 5)),
            BatchNorm2d(num_features=level),
            ReLU(inplace=True)
        )
        self.enc1 = Encoder(level)
        self.enc2 = Encoder(level * 2)
        self.conv1 = Sequential(
            Conv2d(in_channels=level * 4, out_channels=level * 4, kernel_size=(3, 3)),
            BatchNorm2d(num_features=level * 4),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=level * 4, out_channels=level * 4, kernel_size=(3, 3)),
            BatchNorm2d(num_features=level * 4),
            ReLU(inplace=True)
        )
        self.dec1 = Decoder(level * 4)
        self.dec2 = Decoder(level * 2)
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