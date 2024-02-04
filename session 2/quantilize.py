import cv2
import numpy as np

original_image=cv2.imread("fig-1.png")
height=original_image.shape[0]
width=original_image.shape[1]
gray_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
# Quantilize_image=np.zeros([height,width,3],dtype=np.uint8)
img1=gray_image//2*2
img2=gray_image//4*4
img3=gray_image//8*8
img4=gray_image//128*128

cv2.imshow("Original Image",gray_image)
cv2.imshow("Image 1",img1)
cv2.imshow("Image 2",img2)
cv2.imshow("Image 3",img3)
cv2.imshow("Image 4",img4)

cv2.waitKey(0)
    