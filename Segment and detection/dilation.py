import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('clb.png', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

image_dilated = cv2.dilate(image, kernel)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_dilated, cmap='gray')
plt.title('Ảnh sau Dilation')
plt.axis('off')

plt.show()
