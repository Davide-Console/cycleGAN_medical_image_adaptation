# Packages
import multiprocessing

import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dset
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import Functions as Foo

from Models import Generator
from Models import Discriminator

# GENERATE A PICTURE

# Add support for when a program which uses multiprocessing has been frozen to produce a Windows executable
multiprocessing.freeze_support()

# Random seed
torch.manual_seed(123)

# Set device
torch.cuda.empty_cache()

''' QUANDO CUDA NON DARA' PROBLEMI:
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")'''

device = torch.device("cpu")

# Load Networks

nc = 1
ndf = 64

G_A2B = Generator().to(device=device)
G_A2B.load_state_dict(torch.load('genCT2MRI.pth'))
G_A2B.eval()

G_B2A = Generator().to(device=device)
G_B2A.load_state_dict(torch.load('genMRI2CT.pth'))
G_B2A.eval()

D_A = Discriminator(nc, ndf).to(device=device)
D_A.load_state_dict(torch.load('discriminatorCT.pth'))
D_A.eval()

D_B = Discriminator(nc, ndf).to(device=device)
D_B.load_state_dict(torch.load('discriminatorMRI.pth'))
D_B.eval()

# load data
bs = 5
workers = 0

dataroot = 'Data/CT/Test/'
dataset_CT_test = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
]))

dataloader_test_CT = torchdata.DataLoader(dataset_CT_test, batch_size=bs, shuffle=True, num_workers=workers)

dataroot = 'Data/MRI/Test/'
dataset_MR_test = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
]))

dataloader_test_MR = torchdata.DataLoader(dataset_CT_test, batch_size=bs, shuffle=True, num_workers=workers)

data_CT = next(iter(dataloader_test_CT))
data_MR = next(iter(dataloader_test_MR))

A_real = data_CT[0].to(device=device)
B_real = data_MR[0].to(device=device)

# generate images
B_fake = G_A2B(A_real)
A_fake = G_B2A(B_real)

# plot images
for i in range(len(A_real)):
    # In the plot:
    # 1st line: Style
    # 2nd line: Starting image
    # 3rd line: Generated image
    sample1 = A_real[i]
    sample2 = B_real[i]
    sample3 = A_fake[i]
    ax = plt.subplot(3, len(A_real), i + 1)
    ax.axis('off')
    Foo.image_viewer(sample1)
    ax = plt.subplot(3, len(A_real), len(A_real) + i + 1)
    ax.axis('off')
    Foo.image_viewer(sample2)
    ax = plt.subplot(3, len(A_real), 2 * len(A_real) + i + 1)
    ax.axis('off')
    Foo.image_viewer(sample3)

plt.show()

for i in range(len(B_real)):
    sample1 = B_real[i]
    sample2 = A_real[i]
    sample3 = B_fake[i]
    ax = plt.subplot(3, len(B_real), i + 1)
    ax.axis('off')
    Foo.image_viewer(sample1)
    ax = plt.subplot(3, len(B_real), len(B_real) + i + 1)
    ax.axis('off')
    Foo.image_viewer(sample2)
    ax = plt.subplot(3, len(B_real), 2 * len(B_real) + i + 1)
    ax.axis('off')
    Foo.image_viewer(sample3)

plt.show()
