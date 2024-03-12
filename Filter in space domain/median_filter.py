import cv2
import numpy as np
import random as r
from matplotlib import pyplot as plt 

def H_filter(image):
    H=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
    image_filted_by_H=cv2.filter2D(image,-1,H)
    return image_filted_by_H

def addSaltGray(image,n): #add salt-&-pepper noise in grayscale image

    k=0
    salt=True
    ih=image.shape[0]
    iw=image.shape[1]
    noisypixels=(ih*iw*n)/100

    for i in range(ih*iw):
        if k<noisypixels:  #keep track of noise level
                if salt==True:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=255
                        salt=False
                else:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=0
                        salt=True
                k+=1
        else:
            break
    return image

def median_filter(img, i, j, kernel_size):
    values = []
    for k in range(max(0, i - kernel_size // 2), min(img.shape[0], i + kernel_size // 2 + 1)):
        for l in range(max(0, j - kernel_size // 2), min(img.shape[1], j + kernel_size // 2 + 1)):
            values.append(img[k, l])
    return np.median(values)

def apply_median_filter(img, kernel_size):
    height, width = img.shape
    filtered_img = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            filtered_img[i, j] = median_filter(img, i, j, kernel_size)
    return filtered_img

# Đọc ảnh
img = cv2.imread('D:\Python_CV\Filter in space domain\hatgao.jpg', cv2.IMREAD_GRAYSCALE)
image=np.copy(img)
salt_image=addSaltGray(image,0.4)
image_filted_by_H=H_filter(salt_image)
# Kích thước bộ lọc (vd: 3x3, 5x5, ...)
kernel_size = 45
lib_median_filter=cv2.medianBlur(salt_image,kernel_size)

# Áp dụng Median Filter
filtered_img = apply_median_filter(salt_image, kernel_size)

# Hiển thị ảnh gốc và ảnh đã lọc
# cv2.imshow('Original Image', img)
# cv2.imshow('Filtered Image', filtered_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.subplot(3,2,1),plt.imshow(img,cmap='gray') 
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,2),plt.imshow(salt_image,cmap='gray') 
plt.title('Salt noise image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,3),plt.imshow(image_filted_by_H,cmap='gray') 
plt.title('image filted by H filter'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,4),plt.imshow(filtered_img,cmap='gray') 
plt.title('Median filter by function'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,5),plt.imshow(lib_median_filter,cmap='gray') 
plt.title('Median filter by OpenCV'), plt.xticks([]), plt.yticks([]) 
plt.show()