import cv2
import math
import numpy as np
import random
from matplotlib import pyplot as plt


def cal(a1, b1, a2, b2, x):
    return a2 + (x-a1) * (b2-a2) / (b1 - a1)


image = cv2.imread('fig-1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
createdImg = np.zeros([gray.shape[0], gray.shape[1], 3], dtype=np.uint8)
L = 256
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i][j] < 3*L/8:
            createdImg[i][j] = cal(0, 3*L/8, 0, L/8, gray[i][j])
        elif gray[i][j] < 5*L/8:
            createdImg[i][j] = cal(
                3*L/8, 5*L/8, L/8, 3*L/8 + (L-1)/2, gray[i][j])
        else:
            createdImg[i][j] = cal(
                5*L/8, L-1,  3*L/8 + (L-1)/2, L-1, gray[i][j])
cv2.imshow("Origial Image ", gray)
cv2.imshow("Created Image ", createdImg)
plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
# plt.hist(createdImg.ravel(), 256, [0, 256])
# plt.show()
cv2.waitKey(0)
cv2.destroyWindow()
