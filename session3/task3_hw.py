import cv2
import math
import numpy as np
import random
from matplotlib import pyplot as plt


def cal(a1, b1, a2, b2, x):
    return a2 + (x-a1) * (b2-a2) / (b1 - a1)


image = cv2.imread("fig-1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
# plt.hist(createdImg.ravel(), 256, [0, 256])
# plt.show()
cv2.waitKey(0)
cv2.destroyWindow()
