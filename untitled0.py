import cv2
from PIL import Image
import matplotlib.pyplot as plt

img_pil = Image.open("Data/MRI/Train/New folder/0up.jpg")
print('Pillow: ', img_pil.mode, img_pil.size)

img = cv2.imread("Data/CT/Train/New folder/0ctup.jpg", cv2.IMREAD_UNCHANGED)
print('OpenCV: ', img.shape)

plt.figure()
plt.imshow(img)