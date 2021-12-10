import torch
import numpy as np
x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)

# forward pass
y_hat=w*x
loss=(y_hat-y)**2
print(loss)
# backwards pass
loss.backward()
print(w.grad)


# gradient descent manually

# f=w*x
f=2*x
X=np.array([1,2,3,4,5], dtype=np.float32)
Y=np.array([2,4,6,8,10], dtype=np.float32)

w=0.0

# model prediction
def forward(x):
    return w*x
# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MES= 1/N * (w*x-y)**2
# dJ/dw=1/N 2x*(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'prediction before training:f(5)= {forward(5):.3f}')

# train
learning_rate=0.01

n_iter=20
for epoch in range(n_iter):
    y_pred=forward(X)
    l=loss(Y, y_pred)
    dw=gradient(X,Y,y_pred)
    w-=dw*learning_rate
    if epoch%2==0:
        print(f'epoch {epoch+1}:w={w:.3f}, loss={l:.8f}')

print(f'prediction after training:f(5)= {forward(5):.3f}')

