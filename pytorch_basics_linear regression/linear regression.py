import torch
import numpy as np

# Training data
# each column represents a feature
inputs=np.array([[73, 67, 43],
                 [91, 88, 64],
                 [87, 134, 58],
                 [102, 43, 37],
                 [69, 96, 70]], dtype='float32')

# Target value
targets=np.array([[56,70],
                  [81,101],
                  [119,133],
                  [22,37],
                  [103,119]], dtype='float32')

# convert np array to tensor
inputs=torch.from_numpy(inputs)
targets=torch.from_numpy(targets)

# initialize the weight matrix and biases

w=torch.randn(2,3,requires_grad=True)
b=torch.randn(2,requires_grad=True)

# Define the model
# @ is the matrix multiplication in pytorch
def model(x):
    return x @ w.t() +b # the vector b is boardcasted as a matrix here

# Generate predictions
preds=model(inputs)
print(preds)

# compare with targets
print(targets)

# Loss function
# Evaluate how well our model is performing
# Mean square error
def mse(t1,t2):
    diff=t1-t2
     # numel is the number of element
    return torch.sum(diff*diff)/diff.numel()


# compute loss
loss=mse(preds,targets)
print(loss)

# compute the gradients
# loss is a quadratic function of w and b
loss.backward() # compute the gradient of each variable with regard to loss function
print(w) # the value of weight
print(w.grad) # d_loss/d_w

# reset the gradients to zero by calling .zero()
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)

# Ready to train the model with gradient decent
with torch.no_grad():
    w-=w.grad*1e-5
    b-=b.grad*1e-5
    w.grad.zero_()
    b.grad.zero_()

# Train for multiple epochs
for i in range(800):
    preds=model(inputs)
    loss=mse(preds,targets)
    loss.backward()
    with torch.no_grad(): # with statement no grade back track
        w-=w.grad*1e-5
        b-=b.grad*1e-5
        w.grad.zero_()
        b.grad.zero_()

# calculate the loss
preds=model(inputs)
loss=mse(preds,targets)
print(loss)
print(preds)
print(targets)













