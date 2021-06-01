#-*- coding:utf-8 -*-
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        print('inputs:',x.shape)
        x = self.conv1(x)
        print('after conv1:',x.shape)
        x = F.relu(x)
        print('after relu:',x.shape)
        x = self.pool1(x)
        print('after pool1:',x.shape)
        x = self.conv2(x)
        print('after conv2:',x.shape)
        x = F.relu(x)
        print('after relu:',x.shape)
        x = self.pool2(x)
        print('after pool2:',x.shape)
        x = torch.flatten(x, 1)
        print('after flatten:', x.shape)
        x = F.relu(self.fc1(x))
        print('after fc1:', x.shape)
        x = F.relu(self.fc2(x))
        print('after fc2:', x.shape)
        x = self.fc3(x)
        print('after fc3:', x.shape)
        return x


class Net_temp(nn.Module):
    def __init__(self):
        super(Net_temp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,   out_channels=128, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    net = Net_temp()
    summary(net, (1,28,28))
