import torch
import torch.nn.init as init
import numpy as np
import torch.nn as nn



class model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(model,self).__init__()
        self.layer1=nn.Linear(input_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,output_size)


    def forward(self, input):
        output=self.layer1(input)
        output=nn.ReLU()(output)
        output=self.layer2(output)
        return output


Model=model(10,10,10)
print(Model.parameters())









