# -*- coding: utf-8 -*-
import click
import logging
import torch
from pathlib import Path
from torch import nn
import torch.nn.functional as F

class ModelNet(nn.Module):
    
    def __init__(self):
        
        super (ModelNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(12*12*32, 2048) #Flattened the conv4 output
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 2)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, stride=2, kernel_size=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, stride=2, kernel_size=2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, stride=2, kernel_size=2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, stride=2, kernel_size=2)

        x = x.view(-1, 12*12*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x) #No activation
        
        return F.log_softmax(x, dim=1)
    