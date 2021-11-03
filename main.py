# Packages
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dset
import torch.utils.data as torchdata
import torchvision.transforms as transforms

from Models import Generator
from Models import Discriminator

import Functions as Foo


# TODO
# sistemare training loop
# sistemare/aggiungere salvataggio finale di modelli
# aggiungere generazione finale di immagini col test set


# Random seed
torch.manual_seed(123)

# Data load
image_size = (256, 256)
bs = 4  # batch size
workers = 0  # sub processes for data loading (per ora mi funziona solo con =0)


dataroot = 'Data/CT/Train/'  # Directory con le immagini
dataset_CT_train = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize(0, 1),
                              ]))  # convert to greyscale and normalize images
# img loading
dataloader_train_CT = torchdata.DataLoader(dataset_CT_train, batch_size=bs, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader_train_CT))  # equivalent to a "for cycle" for batch load

fig = plt.figure()

for i in range(len(real_batch[0])):
    sample = real_batch[0][i]  # Batch[No batch][No img]
    print(sample.shape)
    ax = plt.subplot(1, len(real_batch[0]), i + 1)

    ax.axis('off')
    Foo.image_viewer(sample)

plt.show()

dataroot = 'Data/CT/Test/'
dataset_CT_test = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize(0, 1),
                              ]))

dataloader_test_CT = torchdata.DataLoader(dataset_CT_test, batch_size=bs, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader_test_CT))

for i in range(len(real_batch[0])):
    sample = real_batch[0][i]
    print(sample.shape)
    ax = plt.subplot(1, len(real_batch[0]), i + 1)

    ax.axis('off')
    Foo.image_viewer(sample)

plt.show()

dataroot = 'Data/MRI/Train/'
dataset_MR_train = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize(0, 1),
                              ]))

dataloader_train_MR = torch.utils.data.DataLoader(dataset_MR_train, batch_size=bs, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader_train_MR))  # equivalent to a for cycle for batch load

for i in range(len(real_batch[0])):
    sample = real_batch[0][i]
    print(sample.shape)
    ax = plt.subplot(1, len(real_batch[0]), i + 1)

    ax.axis('off')
    Foo.image_viewer(sample)

plt.show()

dataroot = 'Data/MRI/Test/'
dataset_MR_test = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize(0, 1),
                              ]))

dataloader_test_MR = torch.utils.data.DataLoader(dataset_MR_test, batch_size=bs, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader_test_MR))

for i in range(len(real_batch[0])):
    sample = real_batch[0][i]
    print(sample.shape)
    ax = plt.subplot(1, len(real_batch[0]), i + 1)

    ax.axis('off')
    Foo.image_viewer(sample)

plt.show()

# Models' load
nc = 1  # No channels = 1 (Grayscale)
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
# Beta1 hyperparam for Adam optimizers
# beta1 = 0.5

optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr)

optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr)
optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr)

# Training the models
Foo.Train(num_epochs, bs, G_A2B, G_B2A, optimizer_G_A2B, optimizer_G_B2A, D_A, D_B,
          optimizer_D_A, optimizer_D_B, dataloader_train_CT,
          dataloader_train_MR, criterion_Im)

# GENERATE A PICTURE
