import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pytorch_BasicNeuralNetwork import *

train_data = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_data  = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)



TLN=Two_Layer_Net(784,50,10,1)

for i in range(10000):
    x=train_data[i][0].view(1,784)
    print(x.shape)
    t=torch.zeros(1,10)
    t[0][train_data[i][1]]=1
    print(t)
    W1 = TLN.params['W1']
    print(W1)
    W2 = TLN.params['W2']
    print(W2)
    L=TLN.loss(x,t)
    L.backward()

    print(W1.retain_grad())

    loss_list=[]
    loss_list.append(L)



print(loss_list)









