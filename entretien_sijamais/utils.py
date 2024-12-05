import torch.nn as nn
import torch.nn.functional as F


class LeNet(
    nn.Module
):  # https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch
    def __init__(self, nb_classes, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 500)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, nb_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))  # Apply relu to each output of conv layer.
        x = F.max_pool2d(x, 2, 2)  # Max pooling layer with kernal of 2 and stride of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(
            -1, 3136
        )  # Flatten dynamically based on batch size and remaining dimensions
        # flatten our images to 1D to input it to the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(
            x
        )  # Applying dropout b/t layers which exchange highest parameters. This is a good practice
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# define the dataset
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, X_train, Y_train, transform=None, target_transform=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.Y_train[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class FFN(nn.Module):
    def __init__(self, nb_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(56 * 56 * 3, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, nb_classes)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 56 * 56 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.fc3(x)
        return x

    import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, nb_classes, block=ResidualBlock):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(256, nb_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


import torch


class PermuteChannels:
    def __call__(self, image):
        if (
            isinstance(image, torch.Tensor)
            and image.ndim == 3
            and image.shape[-1] in {1, 3}
        ):  # (H, W, C)
            return image.permute(2, 0, 1)  # Convert (H, W, C) -> (C, H, W)
        return image


class PermuteChannelsinv:
    def __call__(self, image):
        if (
            isinstance(image, torch.Tensor)
            and image.ndim == 3
            and image.shape[-1] in {1, 3}
        ):  # (H, W, C)
            return image.permute(1, 2, 0)  # Convert (H, W, C) -> (C, H, W)
        return image.permute(1, 2, 0)


# create transforms
