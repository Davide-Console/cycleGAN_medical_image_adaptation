import argparse

from PIL import Image
from skimage.transform import resize

from data_utils import *

def main(path):
    dcm_files=[]

    # Convert uint16 .dcm images to uint8 .png images
    print('Searching for DICOM files...')
    for path, subdirs, files in os.walk(path):
        for f in files:
            if is_dicom(os.path.join(path, f)):
                dcm_files.append(os.path.join(path, f))
        for dir in subdirs:
                break
    print('Converting DICOM file to PNG images...')
    for i, filename in enumerate(dcm_files):
        dcm2png(filename)


    # Set all images to 256x256 and delete gantry
    print('Preprocessing PNG images...')
    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
    for i, filename in enumerate(png_files):
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
    parser.add_argument('-p', '--path', default='Data', help='path to the images')
    args = parser.parse_args()
    main(args.path)
