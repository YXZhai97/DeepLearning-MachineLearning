import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# Download the dateset
dataset = MNIST(root='data/', download=True)
print(len(dataset))
# Download test dateset
test_dataset=MNIST(root='data/',train=False)
print(len(test_dataset))

# show the image date with matplotlib

image, label= dataset[0]
plt.imshow(image,cmap='gray')
plt.show()
print('Label:', label)

# Transform the image data into tensor
# The image is converted to 1x28x28 tensor
import torchvision.transforms as transforms
dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

# split the dataset into train and validation dataset

from torch.utils.data import random_split
train_ds, val_ds=random_split(dataset,[5000,1000])

# load the data in baches
from torch.utils.data import DataLoader
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

# Define the training model
# the logistic regression model is similar to linear regression model
import torch.nn as nn
input_size=28*28
num_classes=10

# Logistic regression model
model=nn.Linear(input_size,num_classes)

# reshape the image to a vector






