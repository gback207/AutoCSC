import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import torch.optim as optim
from torch import cuda

print(cuda.is_available())
device = 'cuda' if cuda.is_available() else 'cpu'


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

pre_test_data=torch.from_numpy(ft[b'data'])
pre_test_labels=ft[b'labels']


test_data=torch.zeros(10000,3,32,32)
for i in range(len(pre_test_data)):
    r=pre_test_data[i][0:1024]
    g=pre_test_data[i][1024:2048]
    b=pre_test_data[i][2048:3072]
    r=r.view(32,32)
    g=g.view(32,32)
    b=b.view(32,32)
    test_data[i][0]=r
    test_data[i][1]=g
    test_data[i][2]=b
test_labels=torch.zeros(100,100, dtype=torch.long)
test_batch_data= torch.zeros(100,100,3,32,32)
for i in range(100):
    for j in range(100):
        test_batch_data[i][j]=test_data[i*100+j]
        test_labels[i][j]=pre_test_labels[i*100+j]
batchdict={}
for key in data_dict.keys():
    pre_data= torch.from_numpy(data_dict[key][b'data'])
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
    pre_labels=data_dict[key][b'labels']
    labels=torch.zeros(100,100, dtype=torch.long)
    batch_data= torch.zeros(100,100,3,32,32)
    for i in range(100):
        for j in range(100):
            batch_data[i][j]= data[i*100+j]
            labels[i][j]=pre_labels[i*100+j]
    batchdict[key]={'batch_data':batch_data,'labels':labels}

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


loss_list=[]
accuracy_list=[]

for epoch in range(5):
    for key in data_dict.keys():
        cnn.train()
        for i in range(100):
            optimizer.zero_grad()
            output=cnn(batchdict[key]['batch_data'][i])
            target=batchdict[key]['labels'][i]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(loss.item())
        loss_list.append(loss.item())

        counter=0
        cnn.eval()
        for i in range(100):
            optimizer.zero_grad()
            output = cnn(test_batch_data[i])
            target = test_labels[i]
            for j in range(100):
                if torch.argmax(output[j])==target[j]:
                    counter+=1

        print(counter/10000)
        accuracy_list.append(counter/10000)





torch.save(cnn, r'C:\Users\brian\OneDrive\바탕 화면\한과영 기타\IDEV\AutoCSC\cnn4.pt')


print(loss_list)
print(accuracy_list)