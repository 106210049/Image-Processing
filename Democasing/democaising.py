import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from math import log10,sqrt
def PSNR_cal(Original,compressed):
    mse=np.mean((Original-compressed)**2)
    if(mse==0):
        return 100
    max_pixel=255.0
    psnr=20*log10(max_pixel/sqrt(mse))
    return psnr
img = cv2.imread("D:\Python_CV\Democasing\image.png") 
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

w_R = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/4 # Gauss Filterring
w_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])/4 
w_B = w_R 
blue = cv2.filter2D(blue, -1, w_B) 
green = cv2.filter2D(green, -1, w_G) 
red = cv2.filter2D(red, -1, w_R) 

picture_restored = np.zeros((r, c, 3), dtype=np.uint8) 
picture_restored[:, :, 0] = blue 
picture_restored[:, :, 1] = green 
picture_restored[:, :, 2] = red 
PSNR=PSNR_cal(img,picture_restored)
print("PSNR= ",PSNR)
plt.subplot(3,3,1) 
plt.imshow(img), plt.title('Original'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,3,2) 
plt.imshow(demos_picture), plt.title('Demosic'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,3,3) 
plt.imshow(picture_restored), plt.title('Restored'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,3,4) 
plt.imshow(picture_restored[:, :, 0]), plt.title('Blue restored'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5) 
plt.imshow(picture_restored[:, :, 1]), plt.title('Green restored'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6) 
plt.imshow(picture_restored[:, :, 2]), plt.title('Red restored'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7) 
plt.imshow(demos_picture[:, :, 0]), plt.title('Blue'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8) 
plt.imshow(demos_picture[:, :, 1]), plt.title('Green'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,9) 
plt.imshow(demos_picture[:, :, 2]), plt.title('Red'), plt.xticks([]), plt.yticks([])
plt.show()