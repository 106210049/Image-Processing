import cv2
import numpy as np
import matplotlib.pyplot as plt
def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (row, col, rgb) = img.shape
    mr = np.zeros((row, col))
    mg = np.zeros((row, col))
    mb = np.zeros((row, col))
    mr[::2, ::2] = 1
    mb[1::2, 1::2] = 1
    mg = 1 - mr - mb
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]
    red = np.multiply(red.astype(float), mr)
    green = np.multiply(green.astype(float), mg)
    blue = np.multiply(blue.astype(float), mb)
    multi_img = np.zeros((row, col, 3), dtype=np.uint8)
    multi_img[:, :, 0] = blue.astype(np.uint8)
    multi_img[:, :, 1] = green.astype(np.uint8)
    multi_img[:, :, 2] = red.astype(np.uint8)
    mosaic = red + green + blue
    mosaic_fft = np.fft.fft2(mosaic)
    mosaic_fft_shift = np.fft.fftshift(mosaic_fft)
    mosaic_fft_mag = np.log(np.abs(mosaic_fft_shift))
    plt.subplot(131)
    plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')
    plt.subplot(132)
    plt.imshow(multi_img, cmap='gray'), plt.title('Mosaic'), plt.axis('off')
    plt.subplot(133)
    plt.imshow(mosaic_fft_mag, cmap='gray'), plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.show()
image_paths = [r"xe\xe1.jpg", r"xe\xe2.jpg", r"xe\xe3.jpg", r"xe\xe4.jpg"]
for path in image_paths:
    process_image(path)
