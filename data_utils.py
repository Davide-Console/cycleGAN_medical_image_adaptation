import pydicom as pd
import os
import imageio
import numpy as np
from skimage.filters import threshold_otsu


def dcm2png(filename):
    # Read DICOM data
    dcmfile = pd.dcmread(filename)

    # Check if file is uint16
    if dcmfile.PixelRepresentation != 0:
        raise ValueError('uint16 is required')

    # Read metadata
    metadata = dcmfile.data_element

    # Workflow
    new_image = (225 * dcmfile.pixel_array / dcmfile.pixel_array.max()).astype(np.uint8)

    # Convert DICOM to png file
    new_name = filename+'.png'
    imageio.imwrite(new_name, new_image)
    os.remove(filename)


def gantry_removal(img):
    # Binarize the image using Otsu's method
    mask = img > threshold_otsu(img)

    # Remove noise
    mask_out = np.multiply(img, mask.astype(np.uint8))

    return mask_out


def is_dicom(filepath):
    # Try to read the file using pydicom
    try:
        dcm = pd.dcmread(filepath)
        return True
    except (pd.errors.InvalidDicomError, IsADirectoryError, FileNotFoundError):
        return False