import argparse

from PIL import Image
from skimage.transform import resize

from data_utils import *


def main(path):

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
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-p', '--path', help='path to the images')
    args = parser.parse_args()
    if not args.path:
        path = input('Specify the path to images: ')
        main(path)
        exit()
    main(args.path)
