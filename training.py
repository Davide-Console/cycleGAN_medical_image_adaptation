import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchsummary import summary
from tqdm import tqdm
from training_utils import gradient_penalty, LSGAN_G, image_viewer, score
from main import path_b, path_img, path_models, path_opt

def train(num_epochs, discriminators_epochs, n_batches_train, n_batches_validation, G_A2B, G_B2A, optimizer_G_A2B,
          optimizer_G_B2A, D_A, D_B, optimizer_D_A, optimizer_D_B, Criterion_Im, dataloader_train_CT,
          dataloader_train_PT, dataloader_test_CT, dataloader_test_PT, LAMBDA_GP, device, sp=0, old_score_A=0,
          old_score_B=0):
    """
        Train the model
        :param num_epochs: number of epochs
        :param discriminators_epochs: number of epochs for discriminators
        :param n_batches_train: number of batches for training
        :param n_batches_validation: number of batches for validation
        :param G_A2B: generator A to B
        :param G_B2A: generator B to A
        :param optimizer_G_A2B: optimizer for generator A to B
        :param optimizer_G_B2A: optimizer for generator B to A
        :param D_A: discriminator A
        :param optimizer_D_A: optimizer for discriminator A
        :param optimizer_D_B: optimizer for discriminator A
        :param Criterion_Im
        :param dataloader_train_CT
        :param dataloader_train_PT
        :param dataloader_test_CT
        :param dataloader_test_PT
        :param LAMBDA_GP
        :param device
        :param sp
        :param old_score_A
        :param old_score_B
    """

    iters = 0
    print('\nG_A2B\n', summary((G_A2B), (1, 256, 256)))
    print('\nG_B2A\n', summary((G_B2A), (1, 256, 256)))
    print('\nD_A\n', summary((D_A), (1, 256, 256)))
    print('\nD_B\n', summary((D_B), (1, 256, 256)))
    print('\nStart Training Loop')

    # For each epoch
    for epoch in range(sp, num_epochs):
        print("Epoch: ", epoch + 1)
        G_A2B.train()
        G_B2A.train()
        D_A.train()
        D_B.train()

        # For each batch in the dataloader
        ITER1 = iter(dataloader_train_CT)
        ITER2 = iter(dataloader_train_PT)
        for i in range(n_batches_train):

            data_CT = next(ITER1)
            data_PT = next(ITER2)

            # Set model input
            A_real = data_CT[0].to(device=device)
            B_real = data_PT[0].to(device=device)

            # Generated images using not updated generators
            B_fake = G_A2B(A_real)
            A_rec = G_B2A(B_fake)
            A_fake = G_B2A(B_real)
            B_rec = G_A2B(A_fake)

            for _ in range(discriminators_epochs):
                # Discriminator A training
                # Computes discriminator loss by feeding real A and fake A samples in discriminator A
                optimizer_D_A.zero_grad()  # sets gradient to zero

                gpA = gradient_penalty(D_A, A_real, A_fake, device=device)
                Disc_loss_A = (
                        -(torch.mean(D_A(A_real)) - torch.mean(D_A(A_fake))) + LAMBDA_GP * gpA
                )
                D_A.zero_grad()
                Disc_loss_A.backward(retain_graph=True)
                optimizer_D_A.step()

                # Discriminator B training
                # Computes discriminator loss by feeding real B and fake B samples in discriminator B
                optimizer_D_B.zero_grad()  # sets gradient to zero

                gpB = gradient_penalty(D_B, B_real, B_fake, device=device)
                Disc_loss_B = (
                        -(torch.mean(D_B(B_real)) - torch.mean(D_B(B_fake))) + LAMBDA_GP * gpB
                )
                D_B.zero_grad()
                Disc_loss_B.backward(retain_graph=True)
                optimizer_D_B.step()

            print(
                '[%d/%d][%d/%d]\tDISCRIMINATORS\tLoss_D_A: %.4f\tLoss_D_B: %.4f'
                % (epoch + 1, num_epochs, i + 1, n_batches_train, Disc_loss_A.item(), Disc_loss_B.item()))

            del gpA, gpB
            torch.cuda.empty_cache()

            # least square loss for generators: Loss based on how many samples the discriminator has discovered
            Full_disc_loss_A2B = -torch.mean(D_B(B_fake))
            LSloss_A2B = LSGAN_G(D_B(B_fake))

            # Cycle Consistency: Loss based on how much similar the starting image and the reconstructed images are
            Cycle_loss_A = (Criterion_Im(A_rec, A_real) * 5)
            Cycle_loss_B = (Criterion_Im(B_rec, B_real) * 5)

            # Identity loss: Loss based on how much similar the starting image and the transformed images are
            Identity_loss_A2B = Criterion_Im(G_A2B(B_real), B_real) * 10

            # Backward propagation: computes derivative of loss function based oh weights && Optimization step: values are updated
            Loss_G_A2B = Cycle_loss_A + Identity_loss_A2B + LSloss_A2B + Full_disc_loss_A2B
            optimizer_G_A2B.zero_grad()

            Full_disc_loss_B2A = -torch.mean(D_A(A_fake))
            LSloss_B2A = LSGAN_G(D_A(A_fake))
            Cycle_loss_A = (Criterion_Im(A_rec, A_real) * 5)
            Cycle_loss_B = (Criterion_Im(B_rec, B_real) * 5)
            del A_fake, B_fake
            Identity_loss_B2A = Criterion_Im(G_B2A(A_real), A_real) * 10
            Loss_G_B2A = Cycle_loss_B + Identity_loss_B2A + LSloss_B2A + Full_disc_loss_B2A
            optimizer_G_B2A.zero_grad()

            Loss_G_A2B.backward(retain_graph=True)
            Loss_G_B2A.backward(retain_graph=True)

            optimizer_G_A2B.step()
            optimizer_G_B2A.step()

            G_A2B.zero_grad()
            G_B2A.zero_grad()

            iters += 1

            print(
                '\t\tGENERATORS\tLoss_G_A2B: %.4f\tLoss_G_B2A: %.4f'
                % (Loss_G_A2B, Loss_G_B2A))

            del data_CT, data_PT, A_real, B_real
            torch.cuda.empty_cache()

        if ((epoch + 1) % 1) == 0:
            # load data
            validation_score_A = []
            validation_score_B = []
            ITER1 = iter(dataloader_test_CT)
            ITER2 = iter(dataloader_test_PT)
            print('Start Model Validation')

            for i in tqdm(range(n_batches_validation)):
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

                if i == 1:
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

                    plt.savefig(path_b + 'batches_PET-CT-->CT_{}.png'.format(epoch + 1), cmap=plt.cm.gray)

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

                    plt.savefig(path_b + 'batches_CT-->PET-CT_{}.png'.format(epoch + 1), cmap=plt.cm.gray)

                A_real = A_real.detach().cpu().numpy()
                A_fake = A_fake.detach().cpu().numpy()
                B_real = B_real.detach().cpu().numpy()
                B_fake = B_fake.detach().cpu().numpy()

                # Saving images
                if i == 1:
                    img = np.squeeze(A_fake[7])
                    plt.imsave(path_img + 'photoCT_f_{}.png'.format(epoch + 1), img, cmap=plt.cm.gray)

                    img1 = np.squeeze(B_fake[7])
                    plt.imsave(path_img + 'photoPT_f_{}.png'.format(epoch + 1), img1, cmap=plt.cm.gray)

                    if epoch == 0:
                        img2 = np.squeeze(A_real[7])
                        plt.imsave(path_img + 'photoCT_r_{}.png'.format(epoch + 1), img2, cmap=plt.cm.gray)

                        img3 = np.squeeze(B_real[7])
                        plt.imsave(path_img + 'photoPT_r_{}.png'.format(epoch + 1), img3, cmap=plt.cm.gray)

                final_score_A, final_score_B = score(A_real, A_fake, B_real, B_fake)

                validation_score_A.append(final_score_A)
                validation_score_B.append(final_score_B)

            score_A = np.mean(validation_score_A)
            score_B = np.mean(validation_score_B)
            std_A = np.std(validation_score_A)
            std_B = np.std(validation_score_B)

            print('CT to PET-CT\tmean: ', score_A, '\tstd: ', std_A)
            print('PET-CT to CT\tmean: ', score_B, '\tstd: ', std_B)

            # Saving models current epoch
            torch.save(G_A2B, path_models + 'G_A2B_epoch{}'.format(epoch + 1))
            torch.save(G_B2A, path_models + 'G_B2A_epoch{}'.format(epoch + 1))
            torch.save(D_A, path_models + 'D_A_epoch{}'.format(epoch + 1))
            torch.save(D_B, path_models + 'D_B_epoch{}'.format(epoch + 1))
            torch.save(optimizer_G_A2B.state_dict(), path_opt + 'opt_G_A2B_epoch{}'.format(epoch + 1))
            torch.save(optimizer_G_B2A.state_dict(), path_opt + 'opt_G_B2A_epoch{}'.format(epoch + 1))
            torch.save(optimizer_D_A.state_dict(), path_opt + 'opt_D_A_epoch{}'.format(epoch + 1))
            torch.save(optimizer_D_B.state_dict(), path_opt + 'opt_D_B_epoch{}'.format(epoch + 1))

            # Saving best model
            proposition = ((np.mean([score_A, score_B]) > np.mean([old_score_A, old_score_B])) and (
                        max([abs(score_A - old_score_A), abs(score_B - old_score_B)]) > 3 * max(
                    [abs(score_A - old_score_A), abs(score_B - old_score_B)]) and (
                                    abs(score_A - old_score_A) < 0.03 and abs(score_A - old_score_A) < 0.03)))

            if epoch == 0 or (score_A > old_score_A and score_B > old_score_B) or proposition:
                print('--- -- - NEW BEST MODEL - -- ---')
                torch.save(G_A2B, path_models + 'best_G_A2B')
                torch.save(G_B2A, path_models + 'best_G_B2A')
                torch.save(D_A, path_models + 'best_D_A')
                torch.save(D_B, path_models + 'best_D_B')
                torch.save(optimizer_G_A2B.state_dict(), path_opt + 'best_opt_G_A2B')
                torch.save(optimizer_G_B2A.state_dict(), path_opt + 'best_opt_G_B2A')
                torch.save(optimizer_D_A.state_dict(), path_opt + 'best_opt_D_A')
                torch.save(optimizer_D_B.state_dict(), path_opt + 'best_opt_D_B')

            # Saving scores and other stuff
            old_score_A = score_A
            old_score_B = score_B

            df = pd.read_csv('/content/drive/My Drive/score.csv', header=0)
            df.loc[epoch, 'score A'] = score_A
            df.loc[epoch, 'std A'] = std_A
            df.loc[epoch, 'score B'] = score_B
            df.loc[epoch, 'std B'] = std_B
            df.loc[epoch, 'loss D A'] = Disc_loss_A.detach().cpu().numpy()
            df.loc[epoch, 'loss D B'] = Disc_loss_B.detach().cpu().numpy()
            df.loc[epoch, 'loss G A2B'] = Loss_G_A2B.detach().cpu().numpy()
            df.loc[epoch, 'loss G B2A'] = Loss_G_B2A.detach().cpu().numpy()
            df.loc[epoch, 'cycle loss A'] = Cycle_loss_A.detach().cpu().numpy()
            df.loc[epoch, 'cycle loss B'] = Cycle_loss_B.detach().cpu().numpy()
            df.loc[epoch, 'identity loss A'] = Identity_loss_B2A.detach().cpu().numpy()
            df.loc[epoch, 'identity loss B'] = Identity_loss_A2B.detach().cpu().numpy()
            df.loc[epoch, 'LS loss A'] = LSloss_A2B.detach().cpu().numpy()
            df.loc[epoch, 'LS loss B'] = LSloss_B2A.detach().cpu().numpy()
            df.loc[epoch, 'LS loss B'] = Full_disc_loss_A2B.detach().cpu().numpy()
            df.loc[epoch, 'LS loss B'] = Full_disc_loss_B2A.detach().cpu().numpy()
            df.to_csv('/content/drive/My Drive/score.csv', index=False)

        del A_real, B_real, A_fake, B_fake, sample1, sample2, sample3, Cycle_loss_A, Cycle_loss_B, Identity_loss_B2A, Identity_loss_A2B, Full_disc_loss_A2B, Full_disc_loss_B2A
        torch.cuda.empty_cache()
