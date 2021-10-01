import torch
import numpy as np

#----------------------------------------

# Tensor is a number, vector, matrix or any n dimensional array

# Number t1 is a float number
t1=torch.tensor(4.)

# Vector tensor, each element in the tensor should have the same data type
t2=torch.tensor([1.,2,3,4])

# Matrix tensor, the matrix should have a regular shape 3x3
t3=torch.tensor([[1,2,3.],[4,5,6],[7,8.,9]])

# The shape of tensor
print(t1.shape)
print(t2.shape)
print(t3.shape)
# --------------------------------------------
# Tensor operations and gradients

x=torch.tensor(3.)
w=torch.tensor(4.,requires_grad=True)
b=torch.tensor(5., requires_grad=True)

# Arithmetic Operations
y=w*x +b # y should be 17 now

# compute derivations
y.backward()

# the gradient dy/dx=w=4
print(w.grad)

#-------------------------------------
# create tensor from Numpy
np_x=np.array([[1,2],[3,4.]])
t_y=torch.from_numpy(np_x)
# print(np_x.dtype,t_y.dtype) #float64 torch.float64

# convert a torch tensor to numpy array
np_y= t_y.numpy()
print(np_y)

# 



