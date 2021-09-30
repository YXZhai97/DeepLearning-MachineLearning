import torch

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


