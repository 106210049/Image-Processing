import cv2
import numpy as np

def rotate_image(image,angle):
    height=image.shape[0]
    width=image.shape[1]

    center_x=height//2
    center_y=width//2

    rotation_matrix = cv2.getRotationMatrix2D((center_x,center_y), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

angle=25
original_image=cv2.imread('fig-1.png')
rotated_image=rotate_image(original_image,angle)
cv2.imshow("Original Image",original_image)
cv2.imshow("Rotated Image",rotated_image)
cv2.waitKey()