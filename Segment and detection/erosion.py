# Erosion được sử dụng để thu nhỏ hoặc loại bỏ các đối tượng (object) trên ảnh.
'''
Nó hoạt động bằng cách trượt một cấu trúc hình thang (kernel hoặc structuring element) qua ảnh 
và chỉ giữ lại các pixel có giá trị cao nếu tất cả các pixel trong cấu trúc hình thang đều có giá trị cao.
Nó làm giảm kích thước của đối tượng và loại bỏ các đặc điểm nhỏ.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
image_with_noise = cv2.imread('clb.png')

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image_with_noise, cv2.COLOR_BGR2RGB))
plt.title('Image with Noise')
plt.axis('off')

kernel = np.ones((3, 3), np.uint8)

image_without_noise = cv2.erode(image_with_noise, kernel)

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image_without_noise, cv2.COLOR_BGR2RGB))
plt.title('Image without Noise (after Erosion)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(-image_with_noise + image_without_noise, cv2.COLOR_BGR2RGB))
plt.title('Difference')
plt.axis('off')
plt.show()


# loại bỏ nhiễu trắng
# tách 2 đối tượng được kết nối 