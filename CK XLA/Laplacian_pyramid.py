import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Đọc ảnh Lena 512x512 và chuyển sang ảnh xám
I = cv2.imread('D:\Python_CV\CK XLA\Lena-Soderberg.jpg', cv2.IMREAD_GRAYSCALE)

# Định nghĩa bộ lọc 1D
a = 0.4
h = np.array([0.1, 0.25, 0.4, 0.25, 0.1])
h_transform=h.T
# Tạo bộ lọc 2D từ bộ lọc 1D
W = np.outer(h, h_transform)

# Số mức của Gaussian pyramid
N = 5

# Tạo Gaussian pyramid
G = [I]
for i in range(1, N):
    G.append(cv2.pyrDown(G[i-1]))

# Hiển thị ảnh tại các mức của Gaussian pyramid
for i in range(N):
    plt.subplot(1, N, i+1)
    plt.imshow(G[i], cmap='gray')
    plt.title(f'G{i+1}')
    plt.axis('off')

plt.show()
# Tạo Laplacian pyramid
L = [G[N-1]]  # LN = GN
for i in range(N-1, 0, -1):
    GE = cv2.pyrUp(G[i])
    GE = cv2.resize(GE, (G[i-1].shape[1], G[i-1].shape[0]))  # Resize để cùng kích thước
    L.append(cv2.subtract(G[i-1], GE))
L.reverse()  # Đảo ngược để L1, L2, ..., LN

# Hiển thị ảnh tại các mức của Laplacian pyramid
for i in range(N):
    plt.subplot(1, N, i+1)
    plt.imshow(L[i], cmap='gray')
    plt.title(f'L{i+1}')
    plt.axis('off')

plt.show()
# Khôi phục ảnh gốc từ Laplacian pyramid
I_reconstructed = L[N-1]
for i in range(N-2, -1, -1):
    I_reconstructed = cv2.pyrUp(I_reconstructed)
    I_reconstructed = cv2.resize(I_reconstructed, (L[i].shape[1], L[i].shape[0]))
    I_reconstructed = cv2.add(I_reconstructed, L[i])

# Hiển thị ảnh gốc và ảnh khôi phục
ssim_value = ssim(I, I_reconstructed)
psnr_value=calculate_psnr(I, I_reconstructed)
print(f'ssim= {ssim_value}')
print(f'psnr= {psnr_value}')
print(f'w= {W}')
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(I_reconstructed, cmap='gray')
plt.title('Ảnh khôi phục')
plt.axis('off')

plt.show()
