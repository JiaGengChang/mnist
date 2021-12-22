#!/local/data/public/2021/jgc47/.dl/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #taken from https://nextjournal.com/gkoehler/pytorch-mnist
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #N, 10, 24, 24
        self.conv2 = nn.Conv2d(10,20, kernel_size=5) #N, 20, 12, 12 
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50) #N, 320->50
        self.fc2 = nn.Linear(50, 10) #N, 50->10

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #N, 10, 12, 12
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2)) #N, 20, 4, 4
        x = x.view(-1, 320) # N, 320
        x = F.relu(self.fc1(x)) #N, 320
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) #N, 10
        return F.log_softmax(x,dim=0)
