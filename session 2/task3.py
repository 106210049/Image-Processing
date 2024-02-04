import cv2
import numpy as np
import random
def swap_patches(image, patch_size, position1, position2):
    """
    Hoán đổi hai vùng patch trong ảnh.

    Parameters:
        - image: Ảnh gốc.
        - patch_size: Kích thước của patch (tuple: (height, width)).
        - position1: Vị trí của patch thứ nhất (tuple: (row, col)).
        - position2: Vị trí của patch thứ hai (tuple: (row, col)).

    Returns:
        - Ảnh mới sau khi thực hiện hoán đổi.
    """
    # Sao chép ảnh gốc để tránh ảnh gốc bị thay đổi
    result_image = np.copy(image)

    # Tạo các biến dễ đọc
    row1, col1 = position1
    row2, col2 = position2
    patch_height, patch_width = patch_size

    # Sao chép vùng patch thứ nhất
    patch1 = np.copy(image[row1-patch_height:row1+patch_height, col1-patch_width:col1+patch_width])

    # Hoán đổi vùng patch thứ nhất và thứ hai
    result_image[row1-patch_height:row1+patch_height, col1-patch_width:col1+patch_width] = \
    image[row2-patch_height:row2+patch_height, col2-patch_width:col2+patch_width]
    result_image[row2-patch_height:row2+patch_height, col2-patch_width:col2+patch_width] = patch1

    return result_image

# Đọc ảnh từ tệp tin
image = cv2.imread("fig-1.png")
height=image.shape[0]
width=image.shape[1]
# Kích thước patch và vị trí của hai patch cần hoán đổi
patch_size = (50, 50)
i=random.randint(0,height)
j=random.randint(0,width)
position1 = (i, j)
position2 = (height-i, width-j)

# Thực hiện hoán đổi patch và lưu ảnh mới
result_image = swap_patches(image, patch_size, position1, position2)
cv2.imwrite('result_image.jpg', result_image)
cv2.imshow('Original Image', image)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
