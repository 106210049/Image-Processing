import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread('circuit.jpg', cv2.IMREAD_GRAYSCALE)

# Kích thước của bộ lọc trung bình
kernel_size = 2

# Chiều cao và chiều rộng của ảnh
height, width = img.shape[:2]

# Tạo một ảnh mới để lưu trữ kết quả xử lý
filtered_img = np.zeros((height, width), dtype=np.uint8)

# Hàm tính trung bình trong vùng lân cận
def mean_filter(img, i, j, kernel_size):
    sum = 0
    count = 0
    for k in range(max(0, i - kernel_size // 2), min(height, i + kernel_size // 2 + 1)):
        for l in range(max(0, j - kernel_size // 2), min(width, j + kernel_size // 2 + 1)):
            sum += img[k, l]
            count += 1
    return sum // count

# Áp dụng Mean filter
for i in range(height):
    for j in range(width):
        filtered_img[i, j] = mean_filter(img, i, j, kernel_size)

# Hiển thị ảnh gốc và ảnh đã lọc
cv2.imshow('Original Image', img)
cv2.imshow('Mean Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
