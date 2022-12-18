import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from keras import backend as K
from skimage import filters
from skimage.measure import label
from skimage.metrics import structural_similarity as ssim
from torchsummary import summary
from tqdm import tqdm
import csv
import os


def image_viewer(image):
    """
        image_viewer(image)

        This function takes in a tensor image and displays it.

        Parameters:
        image (tensor): A tensor image.

        Returns:
        None
    """
    image = image.detach().to('cpu').numpy()
    plt.imshow(np.squeeze(np.transpose(image, (1, 2, 0))), cmap=plt.cm.gray)


def LSGAN_D(real, fake):
    """
        This function takes in two tensors, real and fake, and returns the loss
        for the discriminator.

        Parameters
        ----------
        real : torch.Tensor
            The real images.
        fake : torch.Tensor
            The fake images.

        Returns
        -------
        torch.Tensor
            The loss for the discriminator.
    """
    return (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2))


def LSGAN_G(fake):
    """
        This function calculates the loss for the generator in a LSGAN.

        Parameters
        ----------
        fake : torch.Tensor
            The output of the generator.

        Returns
        -------
        torch.Tensor
            The loss for the generator.
    """
    return (torch.mean((fake - 1) ** 2))


def gradient_penalty(critic, real, fake, device="cpu"):
    """
        Calculates the gradient penalty for a batch of images.

        Parameters
        ----------
        critic : torch.nn.Module
            The critic network.
        real : torch.Tensor
            A batch of real images.
        fake : torch.Tensor
            A batch of fake images.
        device : str, optional
            The device to use for calculations.

        Returns
        -------
        torch.Tensor
            The gradient penalty.
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    del alpha
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


def getLargestCC(segmentation):
    """
        Get the largest connected component of a segmentation.

        Parameters
        ----------
        segmentation : ndarray
            The segmentation to be processed.

        Returns
        -------
        largestCC : ndarray
            The largest connected component of the segmentation.
    """
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def get_body_mask(img):
    """
        This function takes in a 512x512 image and returns a 512x512 mask of the body.
        The mask is a binary image where the body is 1 and the background is 0.
        The mask is created by first thresholding the image using Otsu's method.
        Then, the largest connected component is found and set to 1.
        The mask is then inverted and the largest connected component is found again.
        The mask is then inverted again and the largest connected component is set to 0.
        The mask is then returned.
    """
    val = filters.threshold_otsu(img)

    mask = (img > val) * 1.0

    mask = getLargestCC(mask)

    inv_mask = np.zeros((len(mask), len(mask)), dtype=int)
    for x in range(len(inv_mask)):
        for y in range(len(inv_mask)):
            if mask[x, y] == 0:
                inv_mask[x, y] = 1

    inv_mask[0:255, :] = getLargestCC(inv_mask[0:255, :])
    inv_mask[255:511, :] = getLargestCC(inv_mask[255:511, :])

    for x in range(len(inv_mask)):
        for y in range(len(inv_mask)):
            if inv_mask[x, y] == 0:
                mask[x, y] = 1
            elif inv_mask[x, y] == 1:
                mask[x, y] = 0
    return mask


def score(imA_real, imA_fake, imB_real, imB_fake):
    """
        This function calculates the SSIM and DICE coefficient for the given images.
        The SSIM is calculated for the central 192x192 pixels of the image.
        The DICE coefficient is calculated for the whole image.

        Parameters
        ----------
        imA_real : numpy array
            The real image of domain A.
        imA_fake : numpy array
            The fake image of domain A.
        imB_real : numpy array
            The real image of domain B.
        imB_fake : numpy array
            The fake image of domain B.

        Returns
        -------
        score_A : float
            The score for domain
    """
    # SSIM
    simm_CTtoPT_array = []
    simm_PTtoCT_array = []
    for i in range(len(imA_real)):
        imA_real_s = np.squeeze(imA_real[i])
        imA_fake_s = np.squeeze(imA_fake[i])
        imB_real_s = np.squeeze(imB_real[i])
        imB_fake_s = np.squeeze(imB_fake[i])

        simm_CTtoPT_array.append(ssim(imA_real_s[32:224, 32:224], imA_fake_s[32:224, 32:224]))
        simm_PTtoCT_array.append(ssim(imB_real_s[32:224, 32:224], imB_fake_s[32:224, 32:224]))

    simm_CTtoPT = np.mean(simm_CTtoPT_array)
    simm_PTtoCT = np.mean(simm_PTtoCT_array)

    # DICE COEFFICIENT
    dice_coeff_array_A = []
    dice_coeff_array_B = []
    for i in range(len(imA_real)):
        maskA_real = get_body_mask(np.squeeze(imA_real[i]))
        maskA_fake = get_body_mask(np.squeeze(imA_fake[i]))
        maskB_real = get_body_mask(np.squeeze(imB_real[i]))
        maskB_fake = get_body_mask(np.squeeze(imB_fake[i]))

        imA_real_final = maskA_real * 1.0
        imA_fake_final = maskA_fake * 1.0
        imB_real_final = maskB_real * 1.0
        imB_fake_final = maskB_fake * 1.0

        intersection_A = tf.reduce_sum(imA_real_final * imA_fake_final)
        result_A = ((2. * intersection_A + 1) / (K.sum(imA_real_final) + K.sum(imA_fake_final) + 1)).numpy()
        dice_coeff_array_A.append(result_A)

        intersection_B = tf.reduce_sum(imB_real_final * imB_fake_final)
        result_B = ((2. * intersection_B + 1) / (K.sum(imB_real_final) + K.sum(imB_fake_final) + 1)).numpy()
        dice_coeff_array_B.append(result_B)

    dice_coeff_A = np.mean(dice_coeff_array_A)
    dice_coeff_B = np.mean(dice_coeff_array_B)

    score_A = 0.85 * simm_CTtoPT + 0.15 * dice_coeff_A
    score_B = 0.85 * simm_PTtoCT + 0.15 * dice_coeff_B
    return score_A, score_B


def train(path_b, path_img, path_models, path_opt, device,
          num_epochs, discriminators_epochs, n_batches_train, n_batches_validation,
          G_A2B, G_B2A, optimizer_G_A2B, optimizer_G_B2A,
          D_A, D_B, optimizer_D_A, optimizer_D_B,
          Criterion_Im, LAMBDA_GP,
          dataloader_train_CT, dataloader_train_PT, dataloader_test_CT, dataloader_test_PT,
          sp=0, old_score_A=0, old_score_B=0,
          save_figs=True, save_all=True):
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

    os.makedirs(path_models, exist_ok=True)
    os.makedirs(path_opt, exist_ok=True)
    if save_figs:
        os.makedirs(path_img, exist_ok=True)
        os.makedirs(path_b, exist_ok=True)


    fields = ['Epoch', 'std A', 'score B', 'std B', 'loss D A', 'loss D B', 'loss G A2B', 'loss G B2A', 'cycle loss A',
              'cycle loss B', 'identity loss A', 'identity loss B', 'LS loss A', 'LS loss B', 'LS loss B', 'LS loss B']
    with open("score.csv", 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(fields)


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
                'Epoch: [%d/%d] Batch:[%d/%d]\tDISCRIMINATORS\tLoss_D_A: %.4f\tLoss_D_B: %.4f'
                % (epoch + 1, num_epochs, i + 1, n_batches_train, Disc_loss_A.item(), Disc_loss_B.item()))

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
                '\t\t\t\tGENERATORS\tLoss_G_A2B: %.4f\tLoss_G_B2A: %.4f'
                % (Loss_G_A2B, Loss_G_B2A))

            torch.cuda.empty_cache()

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
