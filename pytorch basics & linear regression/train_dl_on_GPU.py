"""
Train DL model with GPU
"""

import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

# get the data ready
dataset = MNIST(root='data/', download=True, transform=ToTensor())
image, label = dataset[555]
print('image.shape:', image.shape)
plt.imshow(image.permute(1, 2, 0), cmap='gray')
plt.show()
print('Label:', label)

# split the data into train and validation
val_size=10000
train_size=len(dataset)-val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
print(len(train_ds), len(val_ds))

# create pytorch dataloader
batch_size=128
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
plt.show()







