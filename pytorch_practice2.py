import torch

x=torch.ones(3,3, requires_grad=True)
print(x)


y=x+2
print(y)

z=y*y
print(z)

'''x.requires_grad_(False)
print(x)
x.requires_grad_(True)
print(x)'''

out=z.mean()

out.backward()
print(x.grad)
y.backward(x)
print(x.grad)
