import cv2
import math
import numpy as np
import random


image = cv2.imread('fig-1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
b = gray / 255.0

# cauA
b1 = b**0.2
b2 = b**0.5
b3 = b**0.8
b4 = b**1.5
b5 = b**2.5
cv2.imshow("Created Image 1", b1)
cv2.imshow("Created Image 2", b2)
cv2.imshow("Created Image 3 ", b3)
cv2.imshow("Created Image 4", b4)
cv2.imshow("Created Image 5", b5)

# cauB


# for i in range(b.shape[0]):
#     for j in range(b.shape[1]):

# Y = math.log(1 + b)
# cv2.imshow("Created Image Y", b1)
# cauC

cv2.waitKey(0)
