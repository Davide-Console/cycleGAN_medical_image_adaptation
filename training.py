import multiprocessing
import argparse

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


def main(args):
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

    # networks HYPERPARAMETERS
    # discriminators
    ndf = 32
    # generators
    f = 32
    blocks = 9
    # loss
    Criterion_Im = nn.L1Loss()

    # Networks' initialization
    channels = 1

    D_A = Discriminator(channels, ndf).to(device=device)
    D_B = Discriminator(channels, ndf).to(device=device)

    G_A2B = Generator(f, blocks).to(device=device)
    G_B2A = Generator(f, blocks).to(device=device)

    # Parameters' initialization
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.learning_rate)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.learning_rate)

    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=args.learning_rate)
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=args.learning_rate)

    # DATA PREPARATION
    dataroot = 'Data/folder1'
    dataset_domain_A = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    dataset_size = len(dataset_domain_A)
    indices = list(range(dataset_size))

    split1 = int(np.floor(args.test_split * dataset_size))
    preparation_indices, test_indices = indices[split1:], indices[:split1]

    split2 = int(np.floor(args.validation_split * len(preparation_indices)))
    train_indices, val_indices = preparation_indices[split2:], preparation_indices[:split2]

    train_dataset_domain_A = Subset(dataset_domain_A, train_indices)
    train_loader_domain_A = torchdata.DataLoader(train_dataset_domain_A, batch_size=args.batch_size, shuffle=True,
                                                 drop_last=True)

    val_dataset_domain_A = Subset(dataset_domain_A, val_indices)
    validation_loader_domain_A = torchdata.DataLoader(val_dataset_domain_A, batch_size=args.batch_size, shuffle=False,
                                                      drop_last=True)

    dataroot = 'Data/folder2'
    dataset_domain_B = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    train_dataset_domain_B = Subset(dataset_domain_B, train_indices)
    train_loader_domain_B = torchdata.DataLoader(train_dataset_domain_B, batch_size=args.batch_size, shuffle=True,
                                                 drop_last=True)

    val_dataset_domain_B = Subset(dataset_domain_B, val_indices)
    validation_loader_domain_B = torchdata.DataLoader(val_dataset_domain_B, batch_size=args.batch_size, shuffle=False,
                                                      drop_last=True)

    print('Trining set split: {:.2f}%\nValidation set split: {:.2f}%\nTest set split: {:.2f}%'
          .format((len(train_indices) / dataset_size) * 100,
                  (len(val_indices) / dataset_size) * 100,
                  (len(test_indices) / dataset_size) * 100))

    # TRAINING MODELS
    n_batches_train = len(train_loader_domain_A)
    n_batches_validation = len(validation_loader_domain_A)

    train(path_b, path_img, path_models, path_opt, device,
          args.epochs, args.discriminators_epochs, n_batches_train, n_batches_validation,
          G_A2B, G_B2A, optimizer_G_A2B, optimizer_G_B2A,
          D_A, D_B, optimizer_D_A, optimizer_D_B,
          Criterion_Im, args.lambda_gp,
          train_loader_domain_A, train_loader_domain_B, validation_loader_domain_A, validation_loader_domain_B,
          save_figs=args.save_figs, save_all=args.save_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-de', '--discriminators_epochs', type=int, default=5,
                        help='number of epochs for the discriminator')
    parser.add_argument('-lgp', '--lambda_gp', type=int, default=5, help='hyperparameter LAMBDA for gradient penalty')
    parser.add_argument('-ts', '--test_split', type=float, default=0.1,
                        help='test/train split. Number correspont to (test set/dataset)')
    parser.add_argument('-vs', '--validation_split', type=float, default=0.2,
                        help='validation/train split. Number correspont to (validation set/train set)')
    parser.add_argument('-sf', '--save_figs', type=bool, default=False,
                        help='If True, saves some figures for each epoch')
    parser.add_argument('-sa', '--save_all', type=bool, default=False,
                        help='If True, saves models and optimizers for each epoch')
    args = parser.parse_args()

    main(args)
