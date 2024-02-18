import cv2
import numpy as np 
import matplotlib.pyplot as plt 
def Find_histogram(image):
    hist=np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i,j]]+=1
    return hist

image=cv2.imread("circuit.jpg",0)
hist=Find_histogram(image)
plt.subplot(1,2,1)
plt.plot(hist)
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.subplot(1,2,2)
plt.imshow(image,cmap="gray")
plt.show()
cv2.waitKey(0)
