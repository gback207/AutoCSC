from __future__ import print_function
import torch

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

