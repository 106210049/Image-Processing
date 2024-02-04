import cv2
import math
import numpy as np


def creatImage2():
    height = 600
    width = 800
    image = np.zeros([height, width, 3], dtype=np.uint8)
    midx = height // 2
    midy = width // 2
    for x in range(height):
        for y in range(width):
            if (abs(y-midy) < 50 and abs(x - midx) < 150) or (abs(y-midy) < 150 and abs(x - midx) < 50):
                image[x][y] = [255, 255, 255]
            else:
                image[x][y] = [0, 0, 255]
    return image


a = creatImage2()
cv2.imshow("Created Image", a)

cv2.waitKey(0)
