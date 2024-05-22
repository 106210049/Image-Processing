from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
a = 0.3  # Thử giá trị khác cho a
h = np.array([1/4 - a/2, 1/4, a, 1/4, 1/4 - a/2])
h_transpose = h.T

# Tạo bộ lọc 2D từ bộ lọc 1D
W = np.outer(h, h_transpose)

# Số mức của Laplacian pyramid
N = 5 # Tăng số mức của pyramid

# Tạo Laplacian pyramid từ ảnh gốc
L = [I]
for i in range(1, N):
    GE = cv2.pyrDown(L[i-1], dstsize=(L[i-1].shape[1] // 2, L[i-1].shape[0] // 2))
    L.append(cv2.subtract(L[i-1], cv2.pyrUp(GE, dstsize=(L[i-1].shape[1], L[i-1].shape[0]))))


# Khôi phục ảnh gốc từ Laplacian pyramid
I_reconstructed = L[-1].astype(np.float32)  # Bắt đầu với ảnh từ mức cao nhất của Laplacian pyramid
for i in range(N-2, -1, -1):
    height, width = L[i].shape[:2]
    dstsize = ((width * 2) + (width % 2), (height * 2) + (height % 2))
    I_reconstructed = cv2.pyrUp(I_reconstructed, dstsize=dstsize).astype(np.float32)  # Chuyển đổi kiểu dữ liệu của ảnh khôi phục sang float32
    L_i_float32 = L[i].astype(np.float32)  # Chuyển đổi kiểu dữ liệu của mức Laplacian pyramid sang float32
    I_reconstructed = cv2.add(I_reconstructed[:height, :width], L_i_float32)  # Thực hiện phép cộng với mức Laplacian hiện tại
# Hiển thị ảnh gốc và ảnh khôi phục
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(I_reconstructed, cmap='gray')
plt.title('Ảnh khôi phục')
plt.axis('off')

plt.show()

# Tính toán SSIM giữa ảnh gốc và ảnh khôi phục
ssim_value = ssim(I, I_reconstructed, data_range=I.max() - I.min())
psnr_value=calculate_psnr(I, I_reconstructed)
print(f'SSIM: {ssim_value}')
print(f'PSNR: {psnr_value}')