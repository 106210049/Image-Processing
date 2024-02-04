import cv2
import math
import numpy as np


def transform():
    image = cv2.imread('fig-1.png')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = image.shape[0]
    width = image.shape[1]
    Img = np.zeros([height//2, width//2, 3], dtype=np.uint8)
    for x in range(height//2):
        for y in range(width//2):
            Img[x][y] = image[2*x][2*y]


    Img = image[::2, ::2]
    return Img
    # bit = 7
    # Img1 = image // 2 * 2
    # Img2 = image // 4 * 4
    # Img3 = image // 16 * 16
    # Img5 = image // 32 * 32
    # Img6 = image // 64 * 64
    # Img4 = (image // 128) * 255
    # """  """
Img=transform()
cv2.imshow("Image", Img)
    # cv2.imshow("Created 1 Image", Img1)
    # cv2.imshow("Created 2 Image", Img2)
    # cv2.imshow("Created 3 Image", Img3)
    # cv2.imshow("Created 4 Image", Img4)
    # cv2.imshow("Created 5 Image", Img5)
    # cv2.imshow("Created 6 Image", Img6)
cv2.waitKey(0)


