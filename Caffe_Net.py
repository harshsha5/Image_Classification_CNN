# --------------------------------------------------------
# Written by Harsh Sharma (https://github.com/harshsha5)
# --------------------------------------------------------
from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import ipdb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CaffeNet(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=3,dropout_prob=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(c_dim, 96, 11, stride=4, padding=0)          #Image input size should ideally be 227 X 227
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.nonlinear = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout(dropout_prob)

        self.flat_dim = 6*6*256

        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 4096, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(4096, 4096, 'relu'))
        self.fc3 = nn.Sequential(*get_fc(4096, num_classes, 'none'))      #Note: Change to None/softmax accordingly

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification score in shape of (N, Nc)
        """
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.pool3(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers

if __name__ == '__main__':
    args, device = parse_args()
    main()