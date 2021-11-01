# Packages
#import cv2
#from PIL import Image
import torch
import torch.utils.data as torchdata
#import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from Models import Discriminator
from Models import Generator

#import Functions as F


# Random seed
torch.manual_seed(123)

# Data load
image_size = (256,256)
bs=5 # batch size
workers=0 # sub processes for data loading

dataroot = 'Data/CT/Train/'
dataset_CT_train = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0), (1)),
                              ]))

dataloader_train_CT = torchdata.DataLoader(dataset_CT_train,
                                                  batch_size=bs, shuffle=True, 
                                                  num_workers=workers)

real_batch = next(iter(dataloader_train_CT)) # equivalent to a for cycle for batch load



dataroot = 'Data/CT/Test/'
dataset_CT_test = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0), (1)),
                              ]))

dataloader_test_CT = torchdata.DataLoader(dataset_CT_test,
                                                  batch_size=bs, shuffle=True, 
                                                  num_workers=workers)

real_batch = next(iter(dataloader_test_CT))

dataroot = 'Data/MRI/Train/'
dataset_MR_train = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0), (1)),
                              ]))

dataloader_train_MR = torch.utils.data.DataLoader(dataset_MR_train, 
                                                  batch_size=bs, shuffle=True, 
                                                  num_workers=workers)

real_batch = next(iter(dataloader_train_MR)) # equivalent to a for cycle for batch load

dataroot = 'Data/MRI/Test/'
dataset_MR_test = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0), (1)),
                              ]))

dataloader_test_MR = torch.utils.data.DataLoader(dataset_MR_test, 
                                                  batch_size=bs, shuffle=True, 
                                                  num_workers=workers)

real_batch = next(iter(dataloader_test_MR))

print(real_batch[0].shape)

for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(np.transpose(real_batch[0][i], (1,2,0)), cmap="gray_r")
    plt.show()

# Models load
nc = 1
ndf = 64
D_A = Discriminator(nc, ndf)
D_B = Discriminator(nc, ndf)

G_A2B = Generator()
G_B2A = Generator()

# Parameters initialization
lr = 0.0002
num_epochs = 25

# Loss function
criterion_Im = torch.nn.L1Loss()
print(criterion_Im)
#Beta1 hyperparam for Adam optimizers
#beta1 = 0.5

optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr)

optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr)
optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr)

# Training the models
#F.Train(num_epochs, bs, G_A2B, G_B2A,optimizer_G_A2B, optimizer_G_B2A, D_A, D_B, 
#          optimizer_D_A, optimizer_D_B, dataloader_train_CT, 
#          dataloader_train_MR, criterion_Im, old = True)

# Generate a picture
