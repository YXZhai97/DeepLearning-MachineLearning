
import numpy as np
import torch
import torch.nn as nn

# Input date  (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets values number of (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')
inputs= torch.from_numpy(inputs)
targets=torch.from_numpy(targets)

# Dataset and data loader

from torch.utils.data import TensorDataset

# define training data set
train_ds=TensorDataset(inputs,targets)

from torch.utils.data import DataLoader
# define batch size
batch_size=5
train_dl=DataLoader(train_ds,batch_size,shuffle=True)

# Define the model with nn.Linear
model = nn.Linear(3,2)

# define the loss function with built-in los function

import torch.nn.functional as F

# use the built in mean square error function as loss function
loss_fn=F.mse_loss

# compute the loss
loss=loss_fn(model(inputs), targets)
print(loss)

# gradient descent using stochastic gradient descent
# model parameter is passed to the optimizer to indicate which parameter should be update
# Define optimizer
opt=torch.optim.SGD(model.parameters(),lr=1e-5)

# Train the model
# Generate predictions
# Calculate the loss
# Compute gradients w.r.t the weights and biases
# Adjust the weights by subtracting a small quantity proportional to the gradient
# Reset the gradients to zero

# utility function to train the model

def fit(num_epochs, model, loss_fn, opt, train_dl):

    #Train the model for several epoches
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            pred=model(xb)
            loss=loss_fn(pred,yb)
            loss.backward()
            # update the weight
            opt.step()
            # reset the gradient to zero
            opt.zero_grad()
        if (epoch+1)%10==0:
            print('Epoch [{}/{}], Loss:{:.4f}'.format(epoch+1,num_epochs,loss.item()))

fit(100,model,loss_fn,opt,train_dl)










