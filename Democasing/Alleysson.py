import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from math import log10, sqrt

def PSNR_cal(Original,compressed):
    mse=np.mean((Original-compressed)**2)
    if(mse==0):
        return 100
    max_pixel=255.0
    psnr=20*log10(max_pixel/sqrt(mse))
    return psnr

img = cv2.imread("D:\Python_CV\Democasing\image.png") 
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
F_L = np.array([[-2, 3, -6, 3, -2], 
                [3, 4,  2, 4,  3 ], 
                [-6, 2, 48, 2, -6], 
                [3, 4,  2, 4,  3 ], 
                [-2, 3, -6, 3, -2]]) * (1/64) 
out = red + green + blue 
lum = cv2.filter2D(out, -1, F_L)  #Luminance
multi_chr = out - lum   # Multiplexed Chrominance

# demultiplexing
redd = np.zeros((row, col)) 
greenn = np.zeros((row, col)) 
bluee = np.zeros((row, col)) 
redd[::2, ::2] = multi_chr[::2, ::2] 
bluee[1::2, 1::2] = multi_chr[1::2, 1::2] 
greenn = multi_chr - redd - bluee 
#interpolation
smp_chr = np.zeros((row, col, 3), dtype=np.float64) 
smp_chr[:, :, 0] = redd 
smp_chr[:, :, 1] = greenn 
smp_chr[:, :, 2] = bluee 
wrb = np.array([[1, 2, 1], 
                [2, 4, 2], 
                [1, 2, 1]])/4 
wg = np.array([[0, 1, 0], 
               [1, 4, 1],
               [0, 1, 0]])/4 
redd = cv2.filter2D(redd, -1, wrb) 
greenn = cv2.filter2D(greenn, -1, wg) 
bluee = cv2.filter2D(bluee, -1, wrb) 
chr = np.zeros((row, col, 3), dtype=np.float64) 
chr[:, :, 0] = bluee 
chr[:, :, 1] = greenn 
chr[:, :, 2] = redd 
picture_res = np.zeros((row, col, 3), dtype=np.uint8) 
picture_res[:, :, 0] = np.clip(chr[:, :, 0] + lum, 0, 255) 
picture_res[:, :, 1] = np.clip(chr[:, :, 1] + lum, 0, 255) 
picture_res[:, :, 2] = np.clip(chr[:, :, 2] + lum, 0, 255) 
PSNR=PSNR_cal(img,picture_res)
print("PSNR= ",PSNR)
plt.subplot(2, 3, 1) 
plt.imshow(img), plt.title('Original'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 2) 
plt.imshow(multi_img), plt.title( 
'Mosaic image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 3) 
plt.imshow(picture_res), plt.title( 
'Restored Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 4) 
plt.imshow(lum), plt.title( 
'Luminance Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 5) 
plt.imshow(multi_chr), plt.title( 
'Multi chro Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 6) 
plt.imshow(chr), plt.title( 
'Chrominance Image'), plt.xticks([]), plt.yticks([]) 
plt.show()

# So sánh với phương pháp trước đó: ảnh Kodak
# Đánh giá tiêu chí PSNR và SSIM