from __future__ import print_function
import torch
import torch.nn.init as init

#Tensor
x1= torch.empty(5,3)
print(x1)
#초기화되지 않은 5x3 행렬을 생성


x2=torch.rand(5,3)
print(x2)
#무작위로 초기화된 행렬을 생성

x3= torch.zeros(5,3)
print(x3)
#0으로 채워진 행렬을 생성

x4= torch.tensor([[5.0,4.7,2.3],[1.2,3.5,6.1]])
print(x4)
#데이터로부터 tensor를 직접 생성

print(x1.size())
#행렬의 크기를 구함


#연산
print(x1+x2)
print(torch.add(x1,x2))
print(x1.add(x2))
#덧셈문법

result=torch.empty(5,3)
torch.add(x1,x2,out=result)
print(result)
#결과 tensor를 인자로 제공

print(x1[:,0])
print(x1[0,:])
#indexing 방법

x5=torch.randn(4,4)
y5=x5.view(16)
z5=x5.view(-1,8)
print(x5)
print(y5)
print(z5)
#tensor 형태 변환

x6=torch.randn(1)
y6=x6.item()
print(y6)
#tensor, 숫자 변환

x7=torch.ones(5)
y7= x7.numpy()
print(y7)
z7=torch.from_numpy(y7)
print(z7)
#torch, numpy 변환

x8=init.normal_(torch.FloatTensor(3,3), std=0.1)
print(x8)
y8=init.uniform_(torch.FloatTensor(3,3), a=0, b=9)
print(y8)
z8=init.constant_(torch.FloatTensor(3,3), 35)
print(z8)
#다양한 종류의 텐서 생성

x9=init.normal_(torch.FloatTensor(4,4), std=1)
print(x9)
y9=torch.rand(4,1)
print(y9)
z9=torch.mm(x9,y9)
print(z9)
#행렬곱

