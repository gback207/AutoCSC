import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

loss=nn.MSELoss()
input=Variable(torch.randn(3),requires_grad=True)
print(input)
target=Variable(torch.tensor([0.1,0.6,0.9]))
print(target)
output=loss(input,target)
print(output)
output.backward()
print(input.grad)
#MSELoss는 float tensor를 입력 받아야 함


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
output = loss(input, target)
print(output)
output.backward()
print(input.grad)
#CrossEntropyLoss는 dtype=torch.long인 target을 입력 받아야 함