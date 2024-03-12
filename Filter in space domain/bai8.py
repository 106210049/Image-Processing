import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

def Generate_noisy_image(shape,mean, std_dev):
    gaus_image=np.random.normal(mean,std_dev,(shape[0],shape[1]))

    return gaus_image

img=cv2.imread("D:\Python_CV\Filter in space domain\hatgao.jpg",0) 
image_shape =(img.shape[0],img.shape[1])
mean=0
std_dev=5
gaus_img=Generate_noisy_image(image_shape,mean,std_dev)
I1=img+gaus_img
H=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
Y1=cv2.filter2D(I1,-1,H) 

plt.subplot(1,3,1),plt.imshow(img,cmap='gray') 
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(1,3,2),plt.imshow(Y1,cmap='gray') 
plt.title('Using H1'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(I1,cmap='gray') 
plt.title('Gauss Image'), plt.xticks([]), plt.yticks([])
plt.show() 
