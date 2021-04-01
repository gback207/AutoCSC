import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pytorch_Basicmodel import *
import torch.optim as optim

train_data = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_data  = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

Model=model(784,50,10)
optimizer=optim.SGD(Model.parameters(),lr=0.01)
loss_fn=nn.CrossEntropyLoss()

for i in range(1000):
    optimizer.zero_grad()
    input=train_data[i][0].view(784)
    print(input)
    label=train_data[i][1]
    print(label)
    pre_target=torch.zeros(10)
    pre_target[label]=1
    target=torch.tensor(pre_target, dtype=torch.long)
    print(target)
    output=Model.forward(input)
    print(output)
    loss=loss_fn(output,target)
    if i%100==0:
        print(loss)
    optimizer.step()










