import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import torch.optim as optim


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

f=unpickle(r'C:\Users\brian\Downloads\cifar-10-python (1).tar\cifar-10-batches-py\data_batch_1')


pre_data= torch.from_numpy(f[b'data'])
data=torch.zeros(10000,3,32,32)
for i in range(len(pre_data)):
    r=pre_data[i][0:1024]
    g=pre_data[i][1024:2048]
    b=pre_data[i][2048:3072]
    r=r.view(32,32)
    g=g.view(32,32)
    b=b.view(32,32)
    data[i][0]=r
    data[i][1]=g
    data[i][2]=b


print(data[0].shape)

labels= f[b'labels']


train_data=[]

batch_data= torch.zeros(100,100,3,32,32)
for i in range(100):
    for j in range(100):
        batch_data[i][j]= data[j]

print(batch_data[0].shape)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 5* 5 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01)

cnn.train()
for i in range(len(batch_data[i])):
    optimizer.zero_grad()
    output=cnn.forward(batch_data[i])
    target=batch_data[i][1]
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(loss)


