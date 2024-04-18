import os
import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ConvTranspose2d, ReLU, Module, Sequential
from torch.utils.data import random_split

from MarsDataset import MarsDataset


class Encoder(Module):
    def __init__(self, inputs, kernel_size):
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=inputs, out_channels=inputs, kernel_size=kernel_size),
            BatchNorm2d(num_features=inputs),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=inputs, out_channels=inputs*2, kernel_size=kernel_size),
            BatchNorm2d(num_features=inputs*2),
            ReLU(inplace=True)
        )
        self.pooling = MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling(x)
        return x


class Decoder(Module):
    def __init__(self, inputs, kernel_size):
        super().__init__()
        self.up_conv = ConvTranspose2d(inputs, inputs // 2, kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = Sequential(
            Conv2d(in_channels=inputs // 2, out_channels=inputs // 2, kernel_size=kernel_size),
            BatchNorm2d(num_features=inputs // 2),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=inputs // 2, out_channels=inputs // 2, kernel_size=kernel_size),
            BatchNorm2d(num_features=inputs // 2),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(Module):
    def __init__(self, kernel_size):
        super(UNet, self).__init__()
        self.enc1 = Encoder(105, kernel_size)
        self.enc2 = Encoder(210, kernel_size)
        self.enc3 = Encoder(420, kernel_size)
        self.enc4 = Encoder(840, kernel_size)
        self.conv1 = Sequential(
            Conv2d(in_channels=840, out_channels=840, kernel_size=kernel_size),
            BatchNorm2d(num_features=840),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=840, out_channels=840, kernel_size=kernel_size),
            BatchNorm2d(num_features=840),
            ReLU(inplace=True)
        )
        self.dec1 = Decoder(1680, kernel_size)
        self.dec2 = Decoder(840, kernel_size)
        self.dec3 = Decoder(420, kernel_size)
        self.dec4 = Decoder(210, kernel_size)
        self.conv3 = Sequential(
            Conv2d(in_channels=105, out_channels=105, kernel_size=(1, 1)),
            BatchNorm2d(num_features=105),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        #x = self.enc3(x)
        #x = self.enc4(x)
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.dec1(x)
        #x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.conv3(x)
        return x


DATASET_PATH = os.environ.get("PATH_DATASETS", "data/my27.zarr")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/unet/")
EPOCHS = 15
LEARNING_RATE = 0.01
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
device = torch.device('cuda' if AVAIL_GPUS else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = MarsDataset(path_file=DATASET_PATH, batch_size=BATCH_SIZE)
# train_len = int(len(dataset) * 0.7)
# test_len = len(dataset) - train_len
# train, test = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

model = UNet(kernel_size=(1, 1))
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print("Start Epoch {}.".format(epoch + 1))
    epoch_loss = 0
    last_batch = None
    first = True
    count = 1
    for batch in dataset:
        count += 1
        optimizer.zero_grad()
        prediction = model(batch[0])
        label = batch[1]
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= len(dataset)
    print("Epoch {}. Loss: {:.4f}.".format(epoch + 1, epoch_loss))

torch.save(model.state_dict(), CHECKPOINT_PATH + "model.pt")

# loading model:
# model=UNet(input_size=(1,1))
# model.load_state_dict(torch.load(CHECKPOINT_PATH))
# model.eval()
