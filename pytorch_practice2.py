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



