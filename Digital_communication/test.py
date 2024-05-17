import cv2
import numpy as np
img = cv2.imread("D:\Python_CV\Test Open CV\Digital_communication\original.bmp")
img_resized = cv2.resize(img, (250, 250))
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

# Chuyển ma trận 2 chiều thành vector 1 chiều
img_vector = img_bin.flatten()

count = 0
rlc_code = ""
for i in range(len(img_vector)):
    if i == 0 or img_vector[i] != img_vector[i - 1]:
        if i != 0:
            if img_vector[i - 1] == 0:
                rlc_code += str(count) + 'B'
            else:
                rlc_code += str(count) + 'W'
        count = 1
    else:
        count += 1

if img_vector[-1] == 0:
    rlc_code += str(count) + 'B'
else:
    rlc_code += str(count) + 'W'

decoded_vector = []
i = 0
while i < len(rlc_code):
    count_str = ""
    while i < len(rlc_code) and rlc_code[i].isdigit():
        count_str += rlc_code[i]
        i += 1
    count = int(count_str)
    symbol = rlc_code[i]
    if symbol == 'B':
        decoded_vector.extend([0] * count)
    elif symbol == 'W':
        decoded_vector.extend([255] * count)
    i += 1

# Đưa vector 1 chiều thành ma trận 2 chiều
decoded_matrix = np.array(decoded_vector).reshape(250, 250).astype(np.uint8)

cv2.imshow('Decoded Image', decoded_matrix)
cv2.imwrite("Decoded_Image.bmp", decoded_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()
decoded_matrix = cv2.imread("Decoded_Image.bmp")

decoded_matrix = decoded_matrix.astype(np.uint8)
img_resized = img_resized.astype(np.uint8)
original_size = len(img_vector)
compressed_size = len(rlc_code)
compression_ratio = original_size / (compressed_size / 8)  # Chuyển kích thước về byte
decoded_matrix = decoded_matrix.astype(np.uint8)
psnr_value = cv2.PSNR(decoded_matrix, img_resized)
print('Original size: {} bytes'.format(original_size // 8)) 
print('Compressed size: {} bytes'.format(compressed_size // 8))  
print('Compression ratio: {:.2f}'.format(compression_ratio))
print('PSNR: {:.2f} dB'.format(psnr_value))