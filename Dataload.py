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
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

f1=unpickle(r'C:\Users\brian\OneDrive\바탕 화면\cifar-10-python (1).tar\cifar-10-batches-py\data_batch_1')
f2=unpickle(r"C:\Users\brian\OneDrive\바탕 화면\cifar-10-python (1).tar\cifar-10-batches-py\data_batch_2")
f3=unpickle(r"C:\Users\brian\OneDrive\바탕 화면\cifar-10-python (1).tar\cifar-10-batches-py\data_batch_3")
f4=unpickle(r'C:\Users\brian\OneDrive\바탕 화면\cifar-10-python (1).tar\cifar-10-batches-py\data_batch_4')
f5=unpickle(r'C:\Users\brian\OneDrive\바탕 화면\cifar-10-python (1).tar\cifar-10-batches-py\data_batch_5')

data_dict={'b1':f1, 'b2':f2, 'b3':f3, 'b4':f4, 'b5':f5}

ft=unpickle(r'C:\Users\brian\OneDrive\바탕 화면\cifar-10-python (1).tar\cifar-10-batches-py\test_batch')

class test_data(Dataset):
    def __init__(self):
        self.x_test=ft[b'data']
        self.y_test=ft[b'labels']

    def __getitem__(self, idx):
        x=torch.from_numpy(self.x_test[idx])
        y=self.y_test[idx]
        return x,y

    def __len__(self):
        return len(self.x_test)

TS=test_data()
test_dataloader=DataLoader(TS, batch_size=100, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.bn3= nn.BatchNorm1d(500)
        self.bn4=nn.BatchNorm1d(10)
        self.drop2d=nn.Dropout2d(p=0.15)
        self.drop=nn.Dropout(p=0.15)




    def forward(self, x):
        x = F.relu(self.drop2d(self.bn1(self.conv1(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.drop2d(self.bn2(self.conv2(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.drop(self.bn3(self.fc1(x))))
        x = F.relu(self.drop(self.bn4(self.fc2(x))))
        return x


cnn = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adagrad(cnn.parameters(), lr=0.001)





for epoch in range(1):
    for key in data_dict:
        class cifardata(Dataset):
            def __init__(self):
                self.x_data=data_dict[key][b'data']
                self.y_data=data_dict[key][b'labels']


            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, index):
                x=torch.FloatTensor(self.x_data)[index]
                y=(self.y_data)[index]
                return x,y

        dataset=cifardata()
        dataloader=DataLoader(dataset,batch_size=100,shuffle=True)

        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            x_train=x_train.view(100,3,32,32)

            cnn.train()
            optimizer.zero_grad()
            output = cnn(x_train)
            target = y_train
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(loss.item())


        counter = 0
        cnn.eval()
        for test_batch_idx, test_set in enumerate(test_dataloader):
            x_test, y_test=test_set
            x_test=x_test.view(100,3,32,32)
            pred=cnn(x_test)
            for j in range(100):
                if torch.argmax(pred[j])==y_test[j]:
                    counter+=1
        print(counter/10000)
















