import torch

weights=torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output= (weights*weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # clear the grad each time

optimizer=torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()


