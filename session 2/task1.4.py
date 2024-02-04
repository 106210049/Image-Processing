import cv2
import math
import numpy as np


def creatImage():
    height = 400
    width = 600
    center_x=height//2
    center_y=width//2
    image = np.zeros([height, width, 3], dtype=np.uint8)
    
    for x in range(height):
        for y in range(width):
            if (y-center_y)**2+(x-center_x)**2<=R**2:
                image[x][y] = [0,0,255]
            else:
                image[x][y] = [255, 255, 255]
    return image

R=100
image = creatImage()
cv2.imshow("Japan Flag", image)

cv2.waitKey(0)
