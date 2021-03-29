import torch
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class Two_Layer_Net:
    def __init__(self, input_size, hidden_size, output_size, weight_std):
        self.params={}
        self.params['W1']=init.normal_(torch.FloatTensor(input_size,hidden_size), std=weight_std)
        self.params['W2'] = init.normal_(torch.FloatTensor(hidden_size, output_size), std=weight_std)
        self.params['b1']=torch.zeros(hidden_size)
        self.params['b2']=torch.zeros(output_size)
        for item in self.parms:
            item.requires_grad(True)


    def predict(self,x):
        W1=self.params['W1']
        W2=self.params['W2']
        b1=self.params['b1']
        b2=self.params['b2']

        a1=torch.mm(W1,x)+b1
        z1=F.sigmoid(a1)
        a2=torch.mm(W2,z1)+b2
        y=F.softmax(a2)
        return y

    def loss(self,x,t):
        y=self.predict(x)
        return torch.nn.CrossEntropyLoss(x,t)






