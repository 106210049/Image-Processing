import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('clb.png', cv2.IMREAD_GRAYSCALE)
noise_density = 0.05 
noise = np.random.random(image.shape) < noise_density
noisy_image = np.copy(image)
noisy_image[noise] = 255 

kernel = np.ones((3, 3), np.uint8) 
eroded_image = cv2.erode(noisy_image, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Ảnh với Nhiễu Trắng')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Ảnh Sau Khi Erosion')
plt.axis('off')
plt.show()

image_dilated = cv2.dilate(eroded_image, kernel)
cv2.imshow("dilation", image_dilated)
cv2.waitKey(0)


