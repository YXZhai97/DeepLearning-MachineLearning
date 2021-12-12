
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
x=np.array([2,1,0.1])
outputs=softmax(x)
print("soft max:", outputs)

x=torch.tensor([2,1,0.1])
outputs=torch.softmax(x, dim=0)
print("softmax with torch:", outputs)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# use torch built-in
loss=nn.CrossEntropyLoss()

Y=torch.tensor([2,0,1])
# number of samples * number of class
Y_pred_good=torch.tensor([[0.1,1,2.1],[2,1,0.1],[2,1,0.1]])
Y_pred_bad=torch.tensor([[2.1,1,0.1],[0.1,1,2.1],[0.1,3,0.1]])
l1=loss(Y_pred_good, Y)
l2=loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())



