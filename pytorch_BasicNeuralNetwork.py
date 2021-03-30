import torch
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class Two_Layer_Net:
    def __init__(self, input_size, hidden_size, output_size, weight_std):
        self.params={}
        self.params['W1']=weight_std*torch.randn(input_size, hidden_size, requires_grad=True)
        self.params['W2'] = weight_std*torch.randn(hidden_size, output_size, requires_grad=True)
        self.params['b1']=torch.zeros(hidden_size)
        self.params['b2']=torch.zeros(output_size)



    def predict(self,x):
        W1=self.params['W1']
        W2=self.params['W2']
        b1=self.params['b1']
        b2=self.params['b2']

        a1=torch.mm(x,W1)
        z1=torch.sigmoid(a1)
        a2=torch.mm(z1,W2)
        y=F.softmax(a2)
        return y

    def loss(self,x,t):
        y=self.predict(x)
        a=t-y
        b=a*a
        return b.sum()







