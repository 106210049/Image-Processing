import cv2
import numpy as np

def swap_circular_patches(image, patch_radius, position1, position2):
    
    # Sao chép ảnh gốc để tránh ảnh gốc bị thay đổi
    result_image = np.copy(image)

    # Tạo mặt nạ hình tròn
    circular_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(circular_mask, position1, patch_radius, (255, 255, 255), thickness=-1)

    # Sao chép vùng patch thứ nhất
    patch1 = np.copy(image[position1[0]-patch_radius:position1[0]+patch_radius, position1[1]-patch_radius:position1[1]+patch_radius])

    # Thay đổi vùng patch thứ nhất với vùng patch thứ hai bằng cách sử dụng mặt nạ hình tròn
    result_image[circular_mask != 0] = image[position2[0]-patch_radius:position2[0]+patch_radius, position2[1]-patch_radius:position2[1]+patch_radius][circular_mask != 0]

    # Thay đổi vùng patch thứ hai với vùng patch thứ nhất
    result_image[position2[0]-patch_radius:position2[0]+patch_radius, position2[1]-patch_radius:position2[1]+patch_radius] = patch1

    return result_image

# Đọc ảnh từ tệp tin
image = cv2.imread("fig-1.png")

# Bán kính của patch hình tròn và vị trí của hai patch cần hoán đổi
patch_radius = 10
position1 = (20, 20)
position2 = (40, 40)

# Thực hiện hoán đổi patch và lưu ảnh mới
result_image = swap_circular_patches(image, patch_radius, position1, position2)
cv2.imwrite('result_image_circular.jpg', result_image)
cv2.imshow('Original Image', image)
cv2.imshow('Result Image with Circular Patch Swap', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

""""
import cv2
import numpy as np

def swap_circular_patches(image, center1, radius1, center2, radius2):
    # Sao chép ảnh để tránh thay đổi ảnh gốc
    swapped_image = image.copy()

    # Tạo mask cho hình tròn 1 và lấy vùng patch
    mask1 = np.zeros_like(image)
    cv2.circle(mask1, center1, radius1, (255, 255, 255), thickness=-1)
    patch1 = cv2.bitwise_and(image, mask1)

    # Tạo mask cho hình tròn 2 và lấy vùng patch
    mask2 = np.zeros_like(image)
    cv2.circle(mask2, center2, radius2, (255, 255, 255), thickness=-1)
    patch2 = cv2.bitwise_and(image, mask2)

    # Hoán đổi vùng patch giữa hai hình tròn
    swapped_image[mask1 > 0] = patch2[mask1 > 0]
    swapped_image[mask2 > 0] = patch1[mask2 > 0]

    return swapped_image

# Đọc ảnh từ tệp tin
image = cv2.imread('example.jpg')

# Tọa độ và bán kính của hai hình tròn cần hoán đổi
center1 = (100, 100)
radius1 = 50
center2 = (200, 200)
radius2 = 50

# Thực hiện hoán đổi
result_image = swap_circular_patches(image, center1, radius1, center2, radius2)

# Hiển thị ảnh gốc và ảnh kết quả
cv2.imshow('Original Image', image)
cv2.imshow('Swapped Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""