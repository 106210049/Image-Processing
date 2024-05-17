import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('clb.png', cv2.IMREAD_GRAYSCALE)
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)

erosion_image = cv2.erode(image, kernel)
opening_image = cv2.dilate(erosion_image, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(opening_image, cmap='gray')
plt.title('Ảnh Sau Khi Opening')
plt.axis('off')

plt.show()
