import cv2
import numpy as np
# import matplotlib.pyplot as plt
def Sampling(original_image,size):
    original_image=cv2.resize(original_image,(800,600))
    height=original_image.shape[0]
    width=original_image.shape[1]
    gray_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    resize_image = np.zeros([height//size, width//size, 1], dtype=np.uint8)
    for x in range(height//size):
        for y in range(width//size):
            resize_image[x][y]=gray_image[size*x][size*y]
    resize_image = gray_image[::size, ::size]
    return resize_image

original_image=cv2.imread("fig-1.png")
resize_image1=Sampling(original_image,2)
resize_image2=Sampling(original_image,4)
gray_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image",gray_image)
cv2.imshow("Resize Image 1",resize_image1)
cv2.imshow("Resize Image 2",resize_image2)
cv2.waitKey()