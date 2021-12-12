import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
"""
define a transformer to transform numpy array to torch.tensor 

"""
class WineDataset(Dataset):

    def __init__(self, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features

        self.x = xy[:, 1:] # size [n_samples, n_features]
        self.y = xy[:, [0]]  # size [n_samples, 1]

        self.transform=transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        # define the features and label
        sample=self.x[index], self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample


    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor):
        self.factor=factor

    def __call__(self, sample):
        inputs, target= sample
        inputs *=self.factor
        return inputs, target



dataset= WineDataset(transform=ToTensor())
first_data=dataset[0]
features, labels=first_data
print(features)
print(type(features), type(labels))

# transformations are chained together using Compose
composed=torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset=WineDataset(transform=composed)
first_data=dataset[0]
features, labels=first_data
print(features)
print(type(features), type(labels))
