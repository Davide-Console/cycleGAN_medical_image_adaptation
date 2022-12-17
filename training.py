import multiprocessing

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Subset

from networks import Discriminator, Generator
from training_utils import train

path_img = 'images'
path_b = 'batches'
path_models = 'models'
path_opt = 'optimizers'


def main():
    multiprocessing.freeze_support()

    # Random seed
    torch.manual_seed(123)

    # Set device
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

    # splitting dataset
    val_split = 0.5
    test_split = 0.0

    # HYPERPARAMETERS
    # data set
    batch_size = 2
    # discriminators
    ndf = 32
    # generators
    f = 32
    blocks = 9
    # training loss
    lr = 0.0002
    num_epochs = 1
    discriminators_epochs = 3
    LAMBDA_GP = 10
    Criterion_Im = nn.L1Loss()

    # Networks' initialization
    channels = 1

    D_A = Discriminator(channels, ndf).to(device=device)
    D_B = Discriminator(channels, ndf).to(device=device)

    G_A2B = Generator(f, blocks).to(device=device)
    G_B2A = Generator(f, blocks).to(device=device)

    # Parameters' initialization
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr)

    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr)
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr)

    # DATA PREPARATION
    dataroot = 'Data/folder1'
    dataset_domain_A = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    dataset_size = len(dataset_domain_A)
    indices = list(range(dataset_size))

    split1 = int(np.floor(test_split * dataset_size))
    preparation_indices, test_indices = indices[split1:], indices[:split1]

    split2 = int(np.floor(val_split * len(preparation_indices)))
    train_indices, val_indices = preparation_indices[split2:], preparation_indices[:split2]

    train_dataset_domain_A = Subset(dataset_domain_A, train_indices)
    train_loader_domain_A = torchdata.DataLoader(train_dataset_domain_A, batch_size=batch_size, shuffle=True,
                                                 drop_last=True)

    val_dataset_domain_A = Subset(dataset_domain_A, val_indices)
    validation_loader_domain_A = torchdata.DataLoader(val_dataset_domain_A, batch_size=batch_size, shuffle=False,
                                                      drop_last=True)

    dataroot = 'Data/folder2'
    dataset_domain_B = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    train_dataset_domain_B = Subset(dataset_domain_B, train_indices)
    train_loader_domain_B = torchdata.DataLoader(train_dataset_domain_B, batch_size=batch_size, shuffle=True,
                                                 drop_last=True)

    val_dataset_domain_B = Subset(dataset_domain_B, val_indices)
    validation_loader_domain_B = torchdata.DataLoader(val_dataset_domain_B, batch_size=batch_size, shuffle=False,
                                                      drop_last=True)

    print('Trining set split: {:.2f}%\nValidation set split: {:.2f}%\nTest set split: {:.2f}%'
          .format((len(train_indices) / dataset_size) * 100,
                  (len(val_indices) / dataset_size) * 100,
                  (len(test_indices) / dataset_size) * 100))

    # TRAINING MODELS
    n_batches_train = len(train_loader_domain_A)
    n_batches_validation = len(validation_loader_domain_A)

    train(path_b, path_img, path_models, path_opt, device,
          num_epochs, discriminators_epochs, n_batches_train, n_batches_validation,
          G_A2B, G_B2A, optimizer_G_A2B, optimizer_G_B2A,
          D_A, D_B, optimizer_D_A, optimizer_D_B,
          Criterion_Im, LAMBDA_GP,
          train_loader_domain_A, train_loader_domain_B, validation_loader_domain_A, validation_loader_domain_B,
          save_figs=True, save_all=False)


if __name__ == '__main__':
    main()
