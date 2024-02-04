import cv2
import math
import numpy as np


def creatImage2():
    height = 400
    width = 800
    image = np.zeros([height, width, 3], dtype=np.uint8)
    midx = height // 2
    midy = width // 2
    for x in range(height):
        for y in range(width):
            if (abs(y-midy) > 40 and abs(x - midx) > 40) :
                image[x][y] = [255, 255, 255]
            else:
                image[x][y] = [0, 0, 255]
                # if(abs(x-0<30) or abs(height-x<30) or abs(y-0<30) or abs(width-y)<30):
                #     image[x][y] = [255, 255, 255]
    return image


a = creatImage2()
cv2.imshow("Created Image", a)

cv2.waitKey(0)
