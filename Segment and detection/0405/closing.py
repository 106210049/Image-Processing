import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('clb.png', cv2.IMREAD_GRAYSCALE)

kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)

dilation_image = cv2.dilate(image, kernel)
closing_image = cv2.erode(dilation_image, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(closing_image, cmap='gray')
plt.title('Ảnh Sau Khi Closing')
plt.axis('off')

plt.show()
