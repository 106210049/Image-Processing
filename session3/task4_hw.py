import cv2
from matplotlib import pyplot as plt
import numpy as np
Img = cv2.imread('image/34.jpg')
gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
# equ = cv2.equalizeHist(gray)
# cv2.imshow("Original Image", gray)
# plt.hist(gray.ravel(), 256, [0, 256])
# plt.hist(equ.ravel(), 256, [0, 256])
# plt.show()

count = np.zeros(256, dtype=int)
for i in range(0, gray.shape[0]):
    for j in range(0, gray.shape[1]):
        count[gray[i][j]] += 1
        # print(gray[i][j])
ss = (gray.shape[0]*gray.shape[1])
print(ss)
sum = np.zeros(256, dtype=float)
for k in range(0, 256):
    sum[k] = sum[k-1] + count[k] / ss
for k in range(0, 256):
    sum[k] = round(sum[k]*255)
    print(k, sum[k])
# for k in range(0, 256):
    # plt.plot(range(0, 256), sum)
plt.bar(range(0, 256), sum)
plt.show()
# cv2.imshow("New Image", equ)
cv2.waitKey(0)
# cv2.destroyWindow()
