import cv2
import csv
from google.colab import drive
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil
from skimage import data
from skimage import filters
from skimage.measure import label
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from tempfile import NamedTemporaryFile
import tensorflow as tf
from keras import backend as K
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as foo
from torch.utils.data import Subset
import torchvision.datasets as dset
import torch.utils.data as torchdata
import torchvision.transforms as transforms
from tqdm import tqdm

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

    # print('DICE COEFFICIENT A:', dice_coeff_A)
    # print('DICE COEFFICIENT B:', dice_coeff_B)

    score_A = 0.85 * simm_CTtoPT + 0.15 * dice_coeff_A
    score_B = 0.85 * simm_PTtoCT + 0.15 * dice_coeff_B
    return score_A, score_B
