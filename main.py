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




# TODO
# . aggiungere score
# . aggiungere salvataggio e loading di modelli
# . aggiungere gridsearch (ottimizzatore di hyperparameters). Cercare alternative (alvin grid search?)


def main():
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


    # DATA PREPARATION
    image_size = (256, 256)
    bs = 5  # Batch size
    # batch number = train dataset / batch size
    workers = 0  # sub processes for data loading

    dataroot = 'Data/CT/Train/'  # Directory with train img
    dataset_CT_train = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0,), (1,)),  # (mean,std) --> output = (input - mean) /std
                                  ]))  # convert to greyscale and normalize images
    # image loading
    dataloader_train_CT = torchdata.DataLoader(dataset_CT_train, batch_size=bs, shuffle=True, num_workers=workers)

    real_batch = next(iter(dataloader_train_CT))  # equivalent to a "for cycle" for batch load
    

    for i in range(len(real_batch[0])):
        sample = real_batch[0][i]  # Batch[No batch][No img]
        ax = plt.subplot(1, len(real_batch[0]), i + 1)
        ax.axis('off')
        Foo.image_viewer(sample)

    plt.show()

    dataroot = 'Data/CT/Test/'
    dataset_CT_test = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0,), (1,)),
                                  ]))

    dataloader_test_CT = torchdata.DataLoader(dataset_CT_test, batch_size=bs, shuffle=True, num_workers=workers)

    real_batch = next(iter(dataloader_test_CT))

    for i in range(len(real_batch[0])):
        sample = real_batch[0][i]
        ax = plt.subplot(1, len(real_batch[0]), i + 1)
        ax.axis('off')
        Foo.image_viewer(sample)

    plt.show()

    dataroot = 'Data/MRI/Train/'
    dataset_MR_train = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0,), (1,)),
                                  ]))

    dataloader_train_MR = torch.utils.data.DataLoader(dataset_MR_train, batch_size=bs, shuffle=True, num_workers=workers)

    real_batch = next(iter(dataloader_train_MR))

    for i in range(len(real_batch[0])):
        sample = real_batch[0][i]
        ax = plt.subplot(1, len(real_batch[0]), i + 1)
        ax.axis('off')
        Foo.image_viewer(sample)

    plt.show()

    dataroot = 'Data/MRI/Test/'
    dataset_MR_test = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0,), (1,)),
                                  ]))

    dataloader_test_MR = torch.utils.data.DataLoader(dataset_MR_test, batch_size=bs, shuffle=True, num_workers=workers)

    real_batch = next(iter(dataloader_test_MR))

    for i in range(len(real_batch[0])):
        sample = real_batch[0][i]
        ax = plt.subplot(1, len(real_batch[0]), i + 1)
        ax.axis('off')
        Foo.image_viewer(sample)

    plt.show()


    # TRAINING MODELS
    # Networks' initialization
    nc = 1  # No channels = 1 (Grayscale)
    ndf = 64

    D_A = Discriminator(nc, ndf).to(device=device)
    D_B = Discriminator(nc, ndf).to(device=device)

    G_A2B = Generator().to(device=device)
    G_B2A = Generator().to(device=device)

    # Parameters' initialization
    lr = 0.0002 #size of steps taken by gradient descent
    num_epochs = 5

    # Loss function
    criterion_Im = torch.nn.L1Loss()
    # Beta1 hyperparam for Adam optimizers
    # beta1 = 0.5 #0,9
    #beta2 = 0.999

    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr)

    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr)
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr)

    # Training the models
    Foo.Train(num_epochs, bs, G_A2B, G_B2A, optimizer_G_A2B, optimizer_G_B2A, D_A, D_B,
              optimizer_D_A, optimizer_D_B, dataloader_train_CT,
              dataloader_train_MR, criterion_Im, device)

    # GENERATE A PICTURE
    # load data
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
        ax = plt.subplot(3, len(A_real), 2*len(A_real) + i + 1)
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
        ax = plt.subplot(3, len(B_real), 2*len(B_real) + i + 1)
        ax.axis('off')
        Foo.image_viewer(sample3)

    plt.show()


if __name__ == '__main__':
    main()
