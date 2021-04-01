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
input = torch.randn(1, 5, requires_grad=True)
#input=torch.tensor([-0.0367,  0.0254, -0.1133,  0.0329, -0.0166,  0.0250,  0.0224,  0.0963, 0.0339,  0.1319])
print(input)
target = torch.empty(1, dtype=torch.long ).random_(5)
#target=torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
print(target)
output = loss(input, target)
print(output)
output.backward()
print(input.grad)
#CrossEntropyLoss는 dtype=torch.long인 target을 입력 받아야 함
#inputdl [[1,2,3]]꼴이어야 함(이중 대괄호)