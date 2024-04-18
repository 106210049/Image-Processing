import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

def Processing(image_path):
    img = cv2.imread(image_path) 
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    (r, c, _) = img.shape 
    mr = np.zeros((r, c)) 
    mg = np.zeros((r, c)) 
    mb = np.zeros((r, c)) 
    mr[::2, ::2] = 1 
    mb[1::2, 1::2] = 1 
    mg = 1 - mb - mr 
    I_red = img[:, :, 2].astype(np.float64) 
    I_green = img[:, :, 1].astype(np.float64) 
    I_blue = img[:, :, 0].astype(np.float64) 
    red = mr * I_red 
    green = mg * I_green 
    blue = mb * I_blue 
    demos_picture = np.zeros((r, c, 3), dtype=np.uint8) 
    demos_picture[:, :, 0] = blue 
    demos_picture[:, :, 1] = green 
    demos_picture[:, :, 2] = red 

    plt.subplot(1,3,1) 
    plt.imshow(img), plt.title('Original'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(1,3,2)
    plt.imshow(demos_picture), plt.title('Mosaic image'), plt.xticks([]), plt.yticks([]) 

    demos_picture=cv2.cvtColor(demos_picture, cv2.COLOR_RGB2GRAY) 
    f1 = np.fft.fft2(demos_picture) 
    f1shift = np.fft.fftshift(f1) 
    magnitude_spectrum1 = 20*np.log(np.abs(f1shift)) 
    plt.subplot(1,3,3), plt.imshow(magnitude_spectrum1, cmap='jet') 
    plt.title('moasic spectrum'), plt.xticks([]), plt.yticks([]) 
    plt.colorbar()
    plt.show() 
image_paths = [r"D:\Python_CV\Frequqnecy_domain\fruit.jpg", r"D:\Python_CV\Frequqnecy_domain\forest.jpg", r"D:\Python_CV\Frequqnecy_domain\beach.jpg", r"D:\Python_CV\Frequqnecy_domain\walking.jpg",r"D:\Python_CV\Frequqnecy_domain\interior.jpg"]
# image_paths="fruit.jpg"
for path in image_paths:
    Processing(path)

