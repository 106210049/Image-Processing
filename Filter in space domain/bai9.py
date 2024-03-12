import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

def Generate_noisy_image(shape,mean, std_dev):
    gaus_image=np.random.normal(mean,std_dev,(shape[0],shape[1]))

    return gaus_image

img = cv2.imread("circuit.jpg", 0) 

# image_shape =(img.shape[0],img.shape[1])
# mean=0
# std_dev=5
# gaus_img=Generate_noisy_image(image_shape,mean,std_dev)

H=np.array([1,2,1])

H1=np.transpose(H)
# câu a 
Y1 = cv2.filter2D(img, -1, H) 
Y2 = cv2.filter2D(img, -1, H1) 
# câu b 
H2=np.abs(Y1)+np.abs(Y2)
Y3 = cv2.filter2D(img, -1, H2) 
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray') 
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 2), plt.imshow(Y1, cmap='gray') 
plt.title('Y1'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 3), plt.imshow(Y2, cmap='gray') 
plt.title('Y2=transpose(H)'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 4), plt.imshow(Y3, cmap='gray') 
plt.title('Y3=abs(Y1)+abs(Y2)'), plt.xticks([]), plt.yticks([]) 
plt.show() 