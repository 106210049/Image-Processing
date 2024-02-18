import cv2
import math
import numpy as np

def Gray_Change(original_image):
    height=original_image.shape[0]
    width=original_image.shape[1]
    gray_img=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    gamma_corrected=np.zeros([height,width,1],dtype=np.uint8)
    gamma_corrected = np.array(255*(original_image / 255) ** 2.2, dtype = 'uint8')
    return gamma_corrected

original_image=cv2.imread("D:\Python_CV\session3\sample.jpg")
new_gray_img=Gray_Change(original_image)
gray_img=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image",gray_img)
cv2.imshow("change",new_gray_img)
cv2.waitKey()

