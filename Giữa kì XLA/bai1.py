import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# Đọc ảnh đầu vào
I = cv2.imread("plumage.png", 0)

# Kernel h
h = np.array([[1, 2, 4, 2, 1]]) / 10
h_phay = np.transpose(h)

# Kernel H
H = np.array([[1, 2, 4, 1, 2], [2, 4, 8, 4, 2], [4, 8, 16, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]]) / 100

# câu a
I1 = cv2.filter2D(I, -1, h)
I2 = cv2.filter2D(I1, -1, h_phay)

# câu b
I3 = cv2.filter2D(I, -1, H)

# Biến đổi Fourier của ảnh
f_img = np.fft.fft2(I)
f_img_shifted = np.fft.fftshift(f_img)

# Tính phổ biên độ
magnitude_spectrum = 20 * np.log(np.abs(f_img_shifted))

# Lấy kích thước của ảnh
rows, cols = I.shape

# Tính tần số u, v
u = np.fft.fftfreq(rows)
v = np.fft.fftfreq(cols)

# Vẽ phổ biên độ
plt.subplot(2, 2, 1), plt.imshow(I, cmap='gray')
plt.title('I'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(I1, cmap='gray')
plt.title('I1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(I2, cmap='gray')
plt.title('I2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(I3, cmap='gray')
plt.title('I3'), plt.xticks([]), plt.yticks([])

# Vẽ phổ biên độ
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xlabel('u')
plt.ylabel('v')
plt.xticks([0, cols], [-0.5, 0.5])
plt.yticks([0, rows], [-0.5, 0.5])
plt.colorbar()
plt.show()