import torch
import torch.nn as nn
import numpy as np

cross_entropy_loss=nn.CrossEntropyLoss()
output=torch.Tensor([[0.5,0.2,0.3]])
target=torch.Tensor([1,0,0])
loss=cross_entropy_loss(output,target)
print(loss)
