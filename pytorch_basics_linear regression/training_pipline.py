"""
1. design the model (input, output size, forward pass)
2. construct loss and optimizer
3. training loop
    forward pass
    backward
    update weights
a very simple one layer linear model with pytorch built-in functions
"""

import torch
import torch.nn as nn
X=torch.tensor([[1],[2],[3],[4],[5],[6],[7]], dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8],[10],[12],[14]], dtype=torch.float32)
X_test=torch.tensor([5], dtype=torch.float32)
n_samples, n_features=X.shape
print(n_samples,n_features)

input_size=n_features
output_size=n_features
# model=nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define the layers
        self.lin=nn.Linear(input_dim, output_dim)

    def forward(self,x):
        return self.lin(x)


model=LinearRegression(input_size, output_size)
print(f'prediction before training:f(5)= {model(X_test).item():.3f}')

# train
learning_rate=0.01
n_iter=100

# define the loss from nn module
loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iter):
    y_pred=model(X)
    l=loss(Y, y_pred)
    l.backward() # dl/dw
    optimizer.step()
    optimizer.zero_grad()
    if epoch%1==0:
        [w,b]=model.parameters()
        print(f'epoch {epoch+1}:w={w[0][0].item():.3f}, loss={l:.8f}')

print(f'prediction after training:f(5)= {model(X_test).item():.3f}')
