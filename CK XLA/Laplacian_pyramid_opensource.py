import cv2 as cv
import numpy as np

A = cv.imread('D:\Python_CV\CK XLA\Lena-Soderberg.jpg')
B = cv.imread('orange.png')
assert A is not None, "File could not be read, check with os.path.exists()"
assert B is not None, "File could not be read, check with os.path.exists()"

# Resize images to the same size
rows, cols, dpt = A.shape
B = cv.resize(B, (cols, rows))

# Generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

# Generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

# Generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpA[i])
    GE = cv.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0]))  # Resize to ensure same size
    L = cv.subtract(gpA[i-1], GE)
    lpA.append(L)

# Generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpB[i])
    GE = cv.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0]))  # Resize to ensure same size
    L = cv.subtract(gpB[i-1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# Now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))  # Resize to ensure same size
    ls_ = cv.add(ls_, LS[i])

# Image with direct connecting each half
real = np.hstack((A[:, :cols // 2], B[:, cols // 2:]))

cv.imwrite('Pyramid_blending2.jpg', ls_)
cv.imwrite('Direct_blending.jpg', real)
cv.imshow("Apple Original",A)
cv.imshow("Orange Original",B)
cv.imshow('Pyramid_blending2.jpg', ls_)
cv.imshow('Direct_blending.jpg', real)
cv.waitKey(0)