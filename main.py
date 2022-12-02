import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Subset
from networks import Discriminator, Generator
from training_utils import image_viewer
from training import train

path_img = '/content/drive/My Drive/Images/'
path_b = '/content/drive/My Drive/batches/'
path_models = '/content/drive/My Drive/models/'
path_opt = '/content/drive/My Drive/optimizers/'

def main():

    multiprocessing.freeze_support()

    # Random seed
    torch.manual_seed(123)

    # Set device
    torch.cuda.empty_cache()

    # Cuda activation
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

    # splitting dataset
    train_split = 0.8
    val_split = 0.2
    test_split = 0.1

    # HYPERPARAMETERS
    # data set
    bs = 8
    # discriminators
    ndf = 32
    # generators
    f = 32
    blocks = 9
    # training loss
    lr = 0.0002
    num_epochs = 10
    discriminators_epochs = 5
    LAMBDA_GP = 10
    Criterion_Im = nn.L1Loss()
    # beta1 = 0.5 #0.9
    # beta2 = 0.999 per adam mett dentro

    # Networks' initialization
    nc = 1

    D_A = Discriminator(nc, ndf).to(device=device)
    D_B = Discriminator(nc, ndf).to(device=device)

    G_A2B = Generator(f, blocks).to(device=device)
    G_B2A = Generator(f, blocks).to(device=device)

    # Parameters' initialization
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr)

    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr)
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr)

    # DATA PREPARATION
    image_size = (256, 256)
    workers = 0

    dataroot = '/content/drive/My Drive/dataset256uint8/FINAL_DATAct'  # Directory with ct img
    dataset_CT = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    dataset_size = len(dataset_CT)

    indices = list(range(dataset_size))
    split1 = int(np.floor(test_split * dataset_size))
    preparation_indices, test_indices = indices[split1:], indices[:split1]
    split2 = int(np.floor(val_split * len(preparation_indices)))
    train_indices, val_indices = preparation_indices[split2:], preparation_indices[:split2]

    print('Training set: ', (len(train_indices) / dataset_size) * 100, '%\nValidation set: ',
          (len(val_indices) / dataset_size) * 100, '%\nTest set: ', (len(test_indices) / dataset_size) * 100, '%')

    train_dataset_CT = Subset(dataset_CT, train_indices)
    train_loader_CT = torchdata.DataLoader(train_dataset_CT, batch_size=bs, shuffle=True, drop_last=True)

    val_dataset_CT = Subset(dataset_CT, val_indices)
    validation_loader_CT = torchdata.DataLoader(val_dataset_CT, batch_size=bs, shuffle=False, drop_last=True)

    dataroot = '/content/drive/My Drive/dataset256uint8/FINAL_DATApetct'  # Directory with pet-ct img
    dataset_PT = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    train_dataset_PT = Subset(dataset_PT, train_indices)
    train_loader_PT = torchdata.DataLoader(train_dataset_PT, batch_size=bs, shuffle=True, drop_last=True)

    val_dataset_PT = Subset(dataset_PT, val_indices)
    validation_loader_PT = torchdata.DataLoader(val_dataset_PT, batch_size=bs, shuffle=False, drop_last=True)

    # image loading
    real_batch1 = next(iter(train_dataset_CT))
    real_batch2 = next(iter(train_dataset_PT))

    for i in range(int(len(real_batch1[0]) / 8)):
        sample = real_batch1[0][i]
        ax = plt.subplot(1, 8, i + 1)
        ax.axis('off')
        image_viewer(sample)
    plt.show()

    for i in range(int(len(real_batch2[0]) / 8)):
        sample = real_batch2[0][i]
        ax = plt.subplot(1, 8, i + 1)
        ax.axis('off')
        image_viewer(sample)
    plt.show()

    # TRAINING MODELS
    n_batches_train = len(train_loader_CT)
    n_batches_validation = len(validation_loader_CT)

    train(num_epochs, discriminators_epochs, n_batches_train, n_batches_validation, G_A2B, G_B2A, optimizer_G_A2B,
          optimizer_G_B2A,
          D_A, D_B, optimizer_D_A, optimizer_D_B, Criterion_Im, train_loader_CT,
          train_loader_PT, validation_loader_CT, validation_loader_PT, LAMBDA_GP, device)


if __name__ == '__main__':
    main()
