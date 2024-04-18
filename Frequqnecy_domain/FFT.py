import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate 
from scipy.ndimage import shift 
import math
def rotate_image(image,angle):
    height=image.shape[0]
    width=image.shape[1]

    center_x=height//2
    center_y=width//2

    rotation_matrix = cv2.getRotationMatrix2D((center_x,center_y), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# 1. tìm và hiển thị phổ biên độ của ảnh (dùng lệnh fft2) 
anh1 = cv2.imread("D:\Python_CV\Frequqnecy_domain\images.jpg", 0) 
# anh2 = cv2.imread("D:\Python_CV\Frequqnecy_domain\image2.png", 0) 
# anh3 = cv2.imread("D:\Python_CV\Frequqnecy_domain\image3.png", 0) 
anh2=rotate_image(anh1,90)
anh3=rotate_image(anh1,45)
f1 = np.fft.fft2(anh1) 
f1shift = np.fft.fftshift(f1) 
magnitude_spectrum1 = 20*np.log(np.abs(f1shift)) 
f2 = np.fft.fft2(anh2) 
f2shift = np.fft.fftshift(f2) 
magnitude_spectrum2 = 20*np.log(np.abs(f2shift)) 
plt.subplot(221), plt.imshow(anh1, cmap='gray') 
plt.title('Input Image'), plt.xticks([]), plt.yticks([]) 
plt.colorbar()
plt.subplot(222), plt.imshow(magnitude_spectrum1, cmap='gray') 
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([]) 
plt.colorbar()
plt.subplot(223), plt.imshow(anh2, cmap='gray') 
plt.title('Input Image'), plt.xticks([]), plt.yticks([]) 
plt.colorbar()
plt.subplot(224), plt.imshow(magnitude_spectrum2, cmap='gray') 
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([]) 
plt.colorbar()
plt.show()
