import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image = cv2.imread('D:\Python_CV\Frequqnecy_domain\images.jpg', cv2.IMREAD_GRAYSCALE)
M, N = input_image.shape
FT_img = np.fft.fft2(input_image)
D0 = 50  # tần số cắt
# Bộ lọc
u = np.arange(0, M)
u[int(M / 2):] = u[int(M / 2):] - M
v = np.arange(0, N)
v[int(N / 2):] = v[int(N / 2):] - N

V, U = np.meshgrid(v, u)
D = np.sqrt(U**2 + V**2)

H = np.double(D <= D0)
G = H * FT_img
output_image = np.real(np.fft.ifft2(G))

plt.subplot(2, 1, 1), plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.subplot(2, 1, 2), plt.imshow(output_image, cmap='gray')
plt.title('Output Image')
plt.show()