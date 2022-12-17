from data_utils import *
import imageio
from skimage.transform import resize
import matplotlib.image as mpimg
from PIL import Image
import PIL

def main():
    path = 'Data/24759123/20010101/MR4'

    # Convert uint16 .dcm images to uint8 .png images
    dcm_files = [f for f in os.listdir(path) if is_dicom(path+'/'+f)]
    for i, filename in enumerate(dcm_files):
        print(f'Converting {filename} ({i+1}/{len(dcm_files)})')
        dcm2png(path+'/'+filename)

    # Set all images to 256x256 and delete gantry
    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
    for i, filename in enumerate(png_files):
        print(f'Processing {filename} ({i+1}/{len(png_files)})')

        img = Image.open(path+'/'+filename)
        img = np.array(img)
        img = resize(img, (256, 256))
        img = gantry_removal(img)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path+'/'+filename)


if __name__ == '__main__':
    main()