import cv2
import numpy as np
import matplotlib.pyplot as plt

def inverse_filter(Y, H):
    # Biến đổi Fourier cho ảnh đã lọc và kernel của bộ lọc
    Y_fft = np.fft.fft2(Y)
    H_fft = np.fft.fft2(H, Y.shape)
    
    # Tính phổ nghịch đảo của kernel (chú ý tránh chia cho 0)
    H_fft_inv = np.where(H_fft != 0, 1 / H_fft, 0)
    
    # Áp dụng bộ lọc nghịch đảo bằng cách nhân phổ ảnh đã lọc với phổ nghịch đảo của kernel
    I_fft = Y_fft * H_fft_inv
    
    # Thực hiện biến đổi Fourier ngược để lấy lại ảnh gốc
    I_restored = np.fft.ifft2(I_fft).real
    
    return I_restored

# Đọc ảnh đầu vào
I = cv2.imread("D:\Python_CV\Frequqnecy_domain\kodim02.png", cv2.IMREAD_GRAYSCALE)

# Tạo bộ lọc H (kernel trung bình)
H = np.ones((3, 3)) / 9

# Tạo ảnh đã lọc Y = I * H
Y = cv2.filter2D(I, -1, H)

# Khôi phục ảnh I từ ảnh đã lọc Y và bộ lọc H
I_restored = inverse_filter(Y, H)

# Hiển thị ảnh gốc, ảnh đã lọc và ảnh khôi phục
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Y, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(I_restored, cmap='gray')
plt.title('Restored Image')
plt.axis('off')

plt.show()
