import torch
import numpy as np
import torch.nn.init as init

x=torch.ones(3,3, requires_grad=True)
print(x)


y=x+2
print(y)

z=y*y
print(z)

x.requires_grad_(False)
print(x)
x.requires_grad_(True)
print(x)

out=z.mean()

out.backward()
print(x.grad)


a=torch.rand([3,3], requires_grad=True)
print(a)
b=init.normal_(torch.FloatTensor(3,3), std=1)
print(b)

c=torch.mm(a,b)
print(c)

c.backward(a)
print(a.grad)


input=torch.randn(4,4, requires_grad=True)
W1= torch.rand(4,2, requires_grad=True)
b1= torch.zeros(2)
W2= torch.rand(2,1, requires_grad=True)
b2=torch.zeros(1)

a1=torch.mm(input,W1)+b1
a2=torch.mm(a1,W2)+b2
output=a2.sum()
output.backward()
print(W1.grad)
print(W2.grad)




