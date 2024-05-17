import cv2 
import math 
import numpy as np 
from PIL import Image
import os

import cv2
import numpy as np

def calculate_image_size_from_data(image_data):
    # Đọc dữ liệu hình ảnh từ mảng byte
    try:
        nparr = np.frombuffer(image_data, np.uint32)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Không thể đọc dữ liệu hình ảnh.")
        return
    
    # Lấy kích thước của hình ảnh
    image_size_bytes = len(image_data)
    
    # Chuyển đổi kích thước thành đơn vị KB hoặc MB
    if image_size_bytes < 1024:
        size_str = f"{image_size_bytes} bytes"
    elif image_size_bytes < 1024 * 1024:
        size_str = f"{image_size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{image_size_bytes / (1024 * 1024):.2f} MB"
    
    return size_str

def sample(gray,shape,period): 
     
    newImg = np.zeros([shape[0], shape[1], 1], dtype=np.uint8) 
    for i in range(0, shape[0], period): 
        for j in range(0, shape[1], period): 
            sum = 0 
            cnt = 0 
            for x in range(0, period): 
                for y in range(0, period): 
                    if i+x < shape[0] and j+y < shape[1]: 
                        sum += gray[i+x][j+y] 
                        cnt += 1 
            for x in range(0, period): 
                for y in range(0, period): 
                    if i+x < shape[0] and j+y < shape[1]: 
                        newImg[i+x][j+y] = sum // cnt 
    return newImg

image = cv2.imread('D:\Python_CV\Test Open CV\Digital_communication\Input.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
shape=image.shape
period_1 = 4 
period_2 = 2
period_3 = 8
newImg_1=sample(gray,shape,period_1) 
newImg_2=sample(gray,shape,period_2) 
newImg_3=sample(gray,shape,period_3) 

print(f"Dung lượng: {image.nbytes} bytes")
print(f"Dung lượng: {newImg_1.nbytes} bytes")
print(f"Dung lượng: {newImg_2.nbytes} bytes")
print(f"Dung lượng: {newImg_3.nbytes} bytes")
image_size_1 = calculate_image_size_from_data(image)
print(f"Dung lượng của ảnh là: {image_size_1}")
image_size_2 = calculate_image_size_from_data(newImg_1)
print(f"Dung lượng của ảnh là: {image_size_2}")
cv2.imshow('Init image', image) 
cv2.imshow('An image with a sampling period of 4', newImg_1) 
cv2.imshow('An image with a sampling period of 2', newImg_2) 
cv2.imshow('An image with a sampling period of 8', newImg_3) 

cv2.waitKey(0) 

