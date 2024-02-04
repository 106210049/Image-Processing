import numpy as np
import cv2
import random

def padding(img, radius):
    sz = img.shape
    if len(sz) == 3:
        padding_up = np.zeros([radius, sz[1] + 2*radius, sz[2]], dtype=np.uint8)
        padding_down = np.zeros_like(padding_up, dtype=np.uint8)
        padding_left = np.zeros([sz[0], radius, sz[2]], dtype=np.uint8)
        padding_right = np.zeros_like(padding_left, dtype=np.uint8)
        image = np.vstack([padding_up, np.hstack([padding_left, img, padding_right]), padding_down])
    elif len(sz) == 2:
        padding_up = np.zeros([radius, sz[1] + 2*radius], dtype=np.uint8)
        padding_down = np.zeros_like(padding_up, dtype=np.uint8)
        padding_left = np.zeros([sz[0], radius], dtype=np.uint8)
        padding_right = np.zeros_like(padding_left, dtype=np.uint8)
        image = np.vstack([padding_up, np.hstack([padding_left, img, padding_right]), padding_down])

    return image

def main():
    img = cv2.imread("fig-1.png")
    sz = img.shape
    radius = 10
    px = int(np.floor(np.random.rand() * sz[0]))
    py = int(np.floor(np.random.rand() * sz[1]))

    qx = int(np.floor(np.random.rand() * sz[0]))
    qy = int(np.floor(np.random.rand() * sz[1]))

    img = padding(img, radius)

    for i in range(1, sz[0] + 2 * radius):
        for j in range(1, sz[1] + 2 * radius):
            if (i - px - radius)**2 + (j - py - radius)**2 <= radius**2:
                a = img[i, j, :].copy()
                img[i, j, :] = img[i + qx - px, j + qy - py, :]
                img[i + qx - px, j + qy - py, :] = a

    img = img[radius + 1: radius + sz[0], radius + 1: sz[1], :]
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
