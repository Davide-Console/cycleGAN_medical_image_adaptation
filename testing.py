import csv
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary
from tqdm import tqdm

from training_utils import *

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
    batch_size = 2
    test_split = 0.1

    # loading networks
    #todo loading nets

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

    test_dataset_domain_A = Subset(dataset_domain_A, test_indices)
    test_loader_domain_A = torchdata.DataLoader(test_dataset_domain_A, batch_size=batch_size, shuffle=True,
                                                drop_last=True)

    dataroot = 'Data/folder2'
    dataset_domain_B = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]))

    test_dataset_domain_B = Subset(dataset_domain_B, test_indices)
    test_loader_domain_B = torchdata.DataLoader(test_dataset_domain_B, batch_size=batch_size, shuffle=True,
                                                drop_last=True)

    # TESTING MODELS
    n_batches_test = len(test_loader_domain_A)

    if save_figs:
        os.makedirs(path_img, exist_ok=True)
        os.makedirs(path_b, exist_ok=True)

    iters = 0
    print('\nG_A2B\n', summary((G_A2B), (1, 256, 256)))
    print('\nG_B2A\n', summary((G_B2A), (1, 256, 256)))
    print('\nD_A\n', summary((D_A), (1, 256, 256)))
    print('\nD_B\n', summary((D_B), (1, 256, 256)))

    # load data
    validation_score_A = []
    validation_score_B = []
    ITER1 = iter(test_loader_domain_A)
    ITER2 = iter(test_loader_domain_B)
    print('Start Model Validation')

    for i in tqdm(range(len(test_loader_domain_A))):
        # for data_CTv in dataloader_test_CT:
        data_CTv = next(ITER1)
        data_PTv = next(ITER2)

        A_real = data_CTv[0].to(device=device)
        B_real = data_PTv[0].to(device=device)

        # generate images
        torch.no_grad()
        G_A2B.eval()
        G_B2A.eval()
        B_fake = G_A2B(A_real)
        A_fake = G_B2A(B_real)

        # saving batchs if save_figs is true
        if save_figs and i == 0:
            for k in range(len(A_real)):
                # In the plot:
                # 1st line: Starting image
                # 2nd line: Generated image
                # 3rd line: Original image
                sample1 = B_real[k]
                sample2 = A_fake[k]
                sample3 = A_real[k]
                ax1 = plt.subplot(3, len(A_real), k + 1)
                ax1.axis('off')
                image_viewer(sample1)
                ax2 = plt.subplot(3, len(A_real), len(A_real) + k + 1)
                ax2.axis('off')
                image_viewer(sample2)
                ax3 = plt.subplot(3, len(A_real), 2 * len(A_real) + k + 1)
                ax3.axis('off')
                image_viewer(sample3)

            plt.savefig(path_b + '/batches_PET-CT-->CT_{}.png'.format(epoch + 1))

            for k in range(len(B_real)):
                sample1 = A_real[k]
                sample2 = B_fake[k]
                sample3 = B_real[k]
                ax1 = plt.subplot(3, len(B_real), k + 1)
                ax1.axis('off')
                image_viewer(sample1)
                ax2 = plt.subplot(3, len(B_real), len(B_real) + k + 1)
                ax2.axis('off')
                image_viewer(sample2)
                ax3 = plt.subplot(3, len(B_real), 2 * len(B_real) + k + 1)
                ax3.axis('off')
                image_viewer(sample3)

            plt.savefig(path_b + '/batches_CT-->PET-CT_{}.png'.format(epoch + 1))

        A_real = A_real.detach().cpu().numpy()
        A_fake = A_fake.detach().cpu().numpy()
        B_real = B_real.detach().cpu().numpy()
        B_fake = B_fake.detach().cpu().numpy()

        # Saving images if save_figs is true
        if save_figs and epoch == 0:
            img2 = np.squeeze(A_real[0])
            plt.imsave(path_img + '/photoCT_r_{}.png'.format(epoch + 1), img2, cmap=plt.cm.gray)

            img3 = np.squeeze(B_real[0])
            plt.imsave(path_img + '/photoPT_r_{}.png'.format(epoch + 1), img3, cmap=plt.cm.gray)

        if save_figs and i == 0:
            img = np.squeeze(A_fake[0])
            plt.imsave(path_img + '/photoCT_f_{}.png'.format(epoch + 1), img, cmap=plt.cm.gray)

            img1 = np.squeeze(B_fake[0])
            plt.imsave(path_img + '/photoPT_f_{}.png'.format(epoch + 1), img1, cmap=plt.cm.gray)


        final_score_A, final_score_B = score(A_real, A_fake, B_real, B_fake)

        validation_score_A.append(final_score_A)
        validation_score_B.append(final_score_B)

    score_A = np.mean(validation_score_A)
    score_B = np.mean(validation_score_B)
    std_A = np.std(validation_score_A)
    std_B = np.std(validation_score_B)

    print('CT to PET-CT Score:\tmean: ', score_A, '\tstd: ', std_A)
    print('PET-CT to CT Score:\tmean: ', score_B, '\tstd: ', std_B)

    # Saving models current epoch if save_all is True
    if save_all:
        torch.save(G_A2B, path_models + '/G_A2B_epoch{}'.format(epoch + 1))
        torch.save(G_B2A, path_models + '/G_B2A_epoch{}'.format(epoch + 1))
        torch.save(D_A, path_models + '/D_A_epoch{}'.format(epoch + 1))
        torch.save(D_B, path_models + '/D_B_epoch{}'.format(epoch + 1))
        torch.save(optimizer_G_A2B.state_dict(), path_opt + '/opt_G_A2B_epoch{}'.format(epoch + 1))
        torch.save(optimizer_G_B2A.state_dict(), path_opt + '/opt_G_B2A_epoch{}'.format(epoch + 1))
        torch.save(optimizer_D_A.state_dict(), path_opt + '/opt_D_A_epoch{}'.format(epoch + 1))
        torch.save(optimizer_D_B.state_dict(), path_opt + '/opt_D_B_epoch{}'.format(epoch + 1))

    # Saving best model
    proposition = ((np.mean([score_A, score_B]) > np.mean([old_score_A, old_score_B])) and (
                max([abs(score_A - old_score_A), abs(score_B - old_score_B)]) > 3 * max(
            [abs(score_A - old_score_A), abs(score_B - old_score_B)]) and (
                            abs(score_A - old_score_A) < 0.03 and abs(score_A - old_score_A) < 0.03)))

    if epoch == 0 or (score_A > old_score_A and score_B > old_score_B) or proposition:
        print('--- -- - NEW BEST MODEL - -- ---')
        torch.save(G_A2B, path_models + '/best_G_A2B')
        torch.save(G_B2A, path_models + '/best_G_B2A')
        torch.save(D_A, path_models + '/best_D_A')
        torch.save(D_B, path_models + '/best_D_B')
        torch.save(optimizer_G_A2B.state_dict(), path_opt + '/best_opt_G_A2B')
        torch.save(optimizer_G_B2A.state_dict(), path_opt + '/best_opt_G_B2A')
        torch.save(optimizer_D_A.state_dict(), path_opt + '/best_opt_D_A')
        torch.save(optimizer_D_B.state_dict(), path_opt + '/best_opt_D_B')

    # Saving scores and other stuff
    old_score_A = score_A
    old_score_B = score_B

    fields = [epoch, score_A, std_A, score_B, std_B, Disc_loss_A.detach().cpu().numpy(),
              Disc_loss_B.detach().cpu().numpy(), Loss_G_A2B.detach().cpu().numpy(),
              Loss_G_B2A.detach().cpu().numpy(), Cycle_loss_A.detach().cpu().numpy(),
              Cycle_loss_B.detach().cpu().numpy(), Identity_loss_B2A.detach().cpu().numpy(),
              Identity_loss_A2B.detach().cpu().numpy(), LSloss_A2B.detach().cpu().numpy(),
              LSloss_B2A.detach().cpu().numpy(), Full_disc_loss_A2B.detach().cpu().numpy(),
              Full_disc_loss_B2A.detach().cpu().numpy()]
    with open("score.csv", 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(fields)

    torch.cuda.empty_cache()












if __name__ == '__main__':
    main()

#todo argparse