import torch

X=torch.tensor([1,2,3,4,5,6,7], dtype=torch.float32)
Y=torch.tensor([2,4,6,8,10,12,14], dtype=torch.float32)

w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction
def forward(x):
    return w*x


# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MES= 1/N * (w*x-y)**2
# dJ/dw=1/N 2x*(w*x-y)


print(f'prediction before training:f(5)= {forward(5):.3f}')

# train
learning_rate=0.01
n_iter=15

for epoch in range(n_iter):
    y_pred=forward(X)
    l=loss(Y, y_pred)
    l.backward() # dl/dw
    with torch.no_grad():
        w-=learning_rate*w.grad
    w.grad.zero_()
    if epoch%2==0:
        print(f'epoch {epoch+1}:w={w:.3f}, loss={l:.8f}')

print(f'prediction after training:f(5)= {forward(5):.3f}')
