import cv2
import numpy as np

def count_unique_gray_levels(image_path):
    # Đọc ảnh xám
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Đếm các giá trị cường độ xám duy nhất
    unique_gray_levels = np.unique(image)
    count = len(unique_gray_levels)
    
    return count, unique_gray_levels

# Đường dẫn đến ảnh
image_path = 'D:\Python_CV\Test Open CV\Digital_communication\picture.jpg'

# Đếm các cường độ xám duy nhất
count, unique_gray_levels = count_unique_gray_levels(image_path)
Bits=np.ceil(np.log2(count))
levels=2**Bits
# Hiển thị kết quả
print(f'Số lượng cường độ xám duy nhất trong ảnh: {count}')
print(f'Cường độ xám duy nhất: {unique_gray_levels}')
print(f'Số bit lượng tử hóa: {Bits}')
print(f'Số mức lượng tử hóa: {levels}')


