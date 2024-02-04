import cv2
import math
import numpy as np


def creatImage():
    height = 600
    width = 600
    image = np.zeros([height, width, 3], dtype=np.uint8)
    for x in range(height):
        for y in range(width):
            if y < width//3:
                image[x][y] = [255, 0, 0]
            elif y < 2 * width//3:
                image[x][y] = [255, 255, 255]
            else:
                image[x][y] = [0, 0, 255]
    return image


a = creatImage()
cv2.imshow("Created Image", a)

cv2.waitKey(0)

