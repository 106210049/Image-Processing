import cv2
import numpy as np
import random

def swap_and_pad_circular_patches(image, center1, center2, patch_radius):
    i1, j1 = center1
    i2, j2 = center2
    # Tạo patch P và Q
    patch_P = extract_circular_patch(image, (i1, j1), patch_radius)
    patch_Q = extract_circular_patch(image, (i2, j2), patch_radius)
    # Hoán vị patch P và Q
    image[i1 - patch_radius:i1 + patch_radius + 1,j1 - patch_radius:j1 + patch_radius + 1] = patch_Q

    image[i2 - patch_radius:i2 + patch_radius + 1,j2 - patch_radius:j2 + patch_radius + 1] = patch_P

    return image

def extract_circular_patch(image, center, patch_radius):
    i, j = center
    M, N = image.shape

    # Tạo một mảng chứa toàn bộ patch, và gán giá trị zero ban đầu
    patch = np.zeros((2 * patch_radius + 1, 2 * patch_radius + 1), dtype=image.dtype)

    # Tạo mask hình tròn
    mask = np.zeros_like(patch)
    cv2.circle(mask, (patch_radius, patch_radius), patch_radius, 1, thickness=cv2.FILLED)

    # Lấy giá trị từ ảnh gốc tại vị trí của mask
    patch[mask > 0] = image[max(0, i - patch_radius):min(M, i + patch_radius + 1),max(0, j - patch_radius):min(N, j + patch_radius + 1)][mask > 0]

    return patch

img = cv2.imread("fig-1.png", cv2.IMREAD_GRAYSCALE)

M, N = img.shape

i, j = random.randint(0, M), random.randint(0, N)

# Tính tọa độ B
i_B, j_B = M - i, N - j

patch_radius = 20
# 
# Hoán vị patch P và Q, và thêm zero nếu cần thiết
result_image = swap_and_pad_circular_patches(img.copy(), (i, j), (i_B, j_B), patch_radius)

cv2.imshow("Anh goc", img)
cv2.imshow("Ket qua", result_image)

cv2.waitKey(0)