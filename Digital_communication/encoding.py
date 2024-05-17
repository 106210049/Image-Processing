import cv2
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Đọc ảnh và chuyển sang ảnh xám
I = cv2.imread('D:\Python_CV\Test Open CV\Digital_communication\original.bmp')
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Thay đổi giá trị ảnh xám theo ngưỡng
gray = (gray // 128) * 255

# Hiển thị ảnh xám
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

h, w = gray.shape
st = [''] * h

# Áp dụng RLE
for i in range(h):
    cnt = 1
    for j in range(1, w):
        if gray[i, j] == gray[i, j-1]:
            cnt += 1
        else:
            if gray[i, j-1] > 128:
                st[i] += '{:08b}'.format(cnt) + '11111111'
            else:
                st[i] += '{:08b}'.format(cnt) + '00000000'
            cnt = 1
    if gray[i, w-1] > 128:
        st[i] += '{:08b}'.format(cnt) + '11111111'
    else:
        st[i] += '{:08b}'.format(cnt) + '00000000'

# Tạo ảnh mới từ dữ liệu RLE
img = np.zeros((h, w), dtype=np.uint8)
bitcountencode = 0

for i in range(h):
    x = 0
    for j in range(0, len(st[i]), 16):
        num = int(st[i][j:j+8], 2)
        bitcountencode += 16
        if st[i][j+8:j+16] == '00000000':
            img[i, x:x+num] = 0
        else:
            img[i, x:x+num] = 255
        x += num

# Tính tỷ lệ nén
rate = (w * h) / bitcountencode
print('Compression Ratio:', rate)

# Hiển thị ảnh
cv2.imshow('Reconstructed Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Tính PSNR và SSIM
psnr_value = calculate_psnr(gray, img)
ssim_value = ssim(gray, img, data_range=255)

print('PSNR(R):', psnr_value)
print('SSIM(R):', ssim_value)
