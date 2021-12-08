
import torch
import numpy as np

x=torch.rand(5,3)
y=torch.ones(2,3, dtype=torch.double)
z=torch.tensor([2,2,3,4])
print("size of x:", x.size())
print(x)
print(x[:,0])
print(x.view(3,5)) # reshape the tensor
print(x.size())
print(y)
print(z)

# convert numpy array to tensor

a=torch.ones(5)
b=a.numpy()
print(a)
print(b)
c=torch.from_numpy(b)
print(c)

