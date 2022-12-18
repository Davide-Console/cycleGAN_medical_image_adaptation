import argparse

import torch.utils.data as torchdata
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Subset

from training_utils import *

path_models = 'models/'
path_img = 'images/'

def main(args):
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

    # LOAD NETWORKS

    G_A2B = torch.load(path_models + args.model1).to(device=device)
    G_B2A = torch.load(path_models + args.model2).to(device=device)
    G_A2B.eval()
    G_B2A.eval()

    # DATA PREPARATION
    dataroot = 'Data/folder1'  # Directory with ct img
    dataset_A = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    dataset_size = len(dataset_A)

    indices = list(range(dataset_size))
    split1 = int(np.floor(args.test_split * dataset_size))
    preparation_indices, test_indices = indices[split1:], indices[:split1]

    test_dataset_A = Subset(dataset_A, test_indices)
    test_loader_A = torchdata.DataLoader(test_dataset_A, batch_size=args.batch_size, shuffle=False)

    dataroot = 'Data/folder2'
    dataset_B = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    test_dataset_B = Subset(dataset_B, test_indices)
    test_loader_B = torchdata.DataLoader(test_dataset_B, batch_size=args.batch_size, shuffle=False)

    # generate images
    test_score_A = []
    test_score_B = []

    n_batches_test = len(test_loader_A)

    ITER1 = iter(test_loader_A)
    ITER2 = iter(test_loader_B)
    for i in tqdm(range(n_batches_test)):

        data_A = next(ITER1)
        data_B = next(ITER2)

        A_real = data_A[0].to(device=device)
        B_real = data_B[0].to(device=device)

        B_fake = G_A2B(A_real)
        A_fake = G_B2A(B_real)

        if args.save_figs:
            for k in range(args.batch_size):
                img = np.squeeze(A_fake[0].detach().numpy())
                plt.imsave(path_img + 'pred_{}{}.png'.format(i, k), img, cmap=plt.cm.gray)

                img1 = np.squeeze(B_fake[0].detach().numpy())
                plt.imsave(path_img + 'pred_{}{}.png'.format(i, k), img1, cmap=plt.cm.gray)

        A_real = A_real.detach().cpu().numpy()
        A_fake = A_fake.detach().cpu().numpy()
        B_real = B_real.detach().cpu().numpy()
        B_fake = B_fake.detach().cpu().numpy()

        final_score_A, final_score_B = score(A_real, A_fake, B_real, B_fake)

        test_score_A.append(final_score_A)
        test_score_B.append(final_score_B)

    score_A = np.mean(test_score_A)
    score_B = np.mean(test_score_B)
    std_A = np.std(test_score_A)
    std_B = np.std(test_score_B)

    print('A to PET-A\tmean: ', score_A, '\tstd: ', std_A)
    print('PET-A to A\tmean: ', score_B, '\tstd: ', std_B)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-m1', '--model1', type=str, default='best_G_A2B', help='first model to load')
    parser.add_argument('-m2', '--model2', type=str, default='best_G_B2A', help='first model to load')
    parser.add_argument('-ts', '--test_split', type=float, default=0.1,
                        help='test/train split. Number correspont to (test set/dataset)')
    parser.add_argument('-sf', '--save_figs', type=bool, default=False,
                        help='If True, saves some figures for each epoch')
    args = parser.parse_args()

    main(args)
