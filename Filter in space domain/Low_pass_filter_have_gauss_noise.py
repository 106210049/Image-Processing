
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

def generate_noisy_image(shape, mean, std_dev):
    gaus_image=np.random.normal(mean,std_dev,(shape[0],shape[1]))
    return gaus_image

image = cv2.imread("circuit.jpg", 0) 
mean_val=0
std_val=5
# img = img.astype(np.float64) + 50*np.random.randn(*img.shape) 
gaus_image=generate_noisy_image((image.shape[0],image.shape[1]),mean_val,std_val)
img=image+gaus_image

cv2.imshow(" gauss",img)
cv2.waitKey()
H1=1/9*np.ones((3,3)) 
H2=1/25*np.ones((5,5)) 
H3=1/81*np.ones((9,9)) 
H4=1/69*np.ones((13,13))
H5=1/49*np.ones((7,7))

Y1=cv2.filter2D(img,-1,H1) 
Y2=cv2.filter2D(img,-1,H2)
Y3=cv2.filter2D(img,-1,H3) 
Y4=cv2.filter2D(img,-1,H4)
Y5=cv2.filter2D(img,-1,H5)

plt.subplot(3,2,1),plt.imshow(img,cmap='gray') 
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,2),plt.imshow(Y1,cmap='gray') 
plt.title('Using H1'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,3),plt.imshow(Y2,cmap='gray') 
plt.title('Using H2'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,4),plt.imshow(Y3,cmap='gray') 
plt.title('Using H3'), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,2,5),plt.imshow(Y4,cmap='gray') 
plt.title('Using H4'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(Y5,cmap='gray') 
plt.title('Using H5'), plt.xticks([]), plt.yticks([])
# print(Y1.shape)
# print(Y2.shape)
# print(Y3.shape)
plt.show() 