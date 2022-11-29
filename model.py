import math
import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################################
# Model definition
###########################################################################

class Crepe(nn.Module):

    def __init__(self):
        super().__init__()

        # layer parameters
        self.in_channels = [1, 1024, 128, 128, 128, 256]
        self.out_channels = [1024, 128, 128, 128, 256, 512]
        self.in_features = 2048
        self.out_features = 360 # PITCH_BINS 
        kernel_sizes = [(512, 1), (64, 1), (64, 1), (64, 1), (64, 1), (64, 1)]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        # Layer definitions
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = nn.BatchNorm2d(
            num_features=self.out_channels[0],
            eps=1e-3, 
            momentum=0.99)

        self.conv2 = nn.Conv2d(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = nn.BatchNorm2d(
            num_features=self.out_channels[1],
            eps=1e-3, 
            momentum=0.99)

        self.conv3 = nn.Conv2d(
            in_channels=self.in_channels[2],
            out_channels=self.out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = nn.BatchNorm2d(
            num_features=self.out_channels[2],
            eps=1e-3, 
            momentum=0.99)

        self.conv4 = nn.Conv2d(
            in_channels=self.in_channels[3],
            out_channels=self.out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = nn.BatchNorm2d(
            num_features=self.out_channels[3],
            eps=1e-3, 
            momentum=0.99)

        self.conv5 = nn.Conv2d(
            in_channels=self.in_channels[4],
            out_channels=self.out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = nn.BatchNorm2d(
            num_features=self.out_channels[4],
            eps=1e-3, 
            momentum=0.99)

        self.conv6 = nn.Conv2d(
            in_channels=self.in_channels[5],
            out_channels=self.out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = nn.BatchNorm2d(
            num_features=self.out_channels[5],
            eps=1e-3, 
            momentum=0.99)

        self.classifier = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features)

    def forward(self, x):

        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through 6 layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN, (0, 0, 31, 32))
        x = self.layer(x, self.conv3, self.conv3_BN, (0, 0, 31, 32))
        x = self.layer(x, self.conv4, self.conv4_BN, (0, 0, 31, 32))
        x = self.layer(x, self.conv5, self.conv5_BN, (0, 0, 31, 32))
        x = self.layer(x, self.conv6, self.conv6_BN, (0, 0, 31, 32))

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute logits
        return torch.sigmoid(self.classifier(x))

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def layer(self, x, conv, batch_norm, padding):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))


