import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pytorch_Basicmodel import *
import torch.optim as optim
from torch.autograd import Variable

train_data = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_data  = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

Model=model(784,50,10)
print(Model.parameters())
optimizer=optim.SGD(Model.parameters(),lr=0.01)
loss_fn=nn.CrossEntropyLoss()

for i in range(10000):
    optimizer.zero_grad()
    input=Variable(train_data[i][0].view(784), requires_grad=True)
    input=input.view(1,-1)
    #print(input)
    label=train_data[i][1]
    #print(label)
    target=torch.tensor([label], dtype=torch.long)
    #pre_target[label]=1
    #print(target)
    output=Variable(Model.forward(input), requires_grad=True)
    #print(output)
    loss=loss_fn(output,target)
    loss.backward()
    if i%100==0:
        print(loss)
    optimizer.step()










