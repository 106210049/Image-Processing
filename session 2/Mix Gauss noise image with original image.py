import numpy as np
import cv2

def creatImage(shape):
    R=50
    height = shape[0]
    width = shape[1]
    center_x=height//2
    center_y=width//2
    image = np.zeros([height, width, 3], dtype=np.uint8)
    
    for x in range(height):
        for y in range(width):
            if (y-center_y)**2+(x-center_x)**2<=R**2:
                image[x][y] = [0,0,255]
            else:
                image[x][y] = [255, 255, 255]
    return image

def generate_noisy_image(shape, mean, std_dev):
    # Tạo ảnh đen với kích thước và kiểu dữ liệu mong muốn
    noisy_image = creatImage(shape)

    # Kích thước ảnh
    height, width = shape[:2]

    # Tạo giá trị ngẫu nhiên theo phân phối Gauss
    for i in range(height):
        for j in range(width):
            pixel_value = int(np.random.normal(mean, std_dev))
            # Giới hạn giá trị pixel trong khoảng [0, 255]
            noisy_image[i, j] = np.clip(pixel_value, 0, 255)
    # noisy_image=np.random.normal(0,5,(noisy_image.shape[0],noisy_image.shape[1],3))
    return noisy_image

def Generate_noisy_image_2(shape, mean, std_dev):
    gaus_image=np.random.normal(mean,std_dev,(shape[0],shape[1],3))
    return gaus_image

def calculate_mean_std(image):
    # Chuyển ảnh về dạng grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kích thước ảnh
    height, width = image.shape

    # Tính mean
    mean_value = sum(sum(image)) / (height * width)

    # Tính std
    std_value = (sum(sum((image - mean_value) ** 2)) / (height * width)) ** 0.5

    return mean_value, std_value

def calculate_mean_std_function(image):
    # Chuyển ảnh về kiểu dữ liệu số thực để tính toán
    image_float = image.astype(np.float32) / 255.0

    # Tính mean và std cho từng kênh màu
    mean_values = np.mean(image_float, axis=(0, 1))
    std_values = np.std(image_float, axis=(0, 1))

    return mean_values, std_values

def calculate_mean_std_color(image):
    # Kích thước ảnh
    height, width, channels = image.shape
    # height=image.shape[0]
    # width=image.shape[1]
    # channels=image.shape[2]

    # Tính mean cho từng kênh màu
    mean_values = [0.0, 0.0, 0.0]
    for i in range(channels):
        mean_values[i] = sum(sum(image[:, :, i])) / (height * width)

    # Tính std cho từng kênh màu
    std_values = [0.0, 0.0, 0.0]
    for i in range(channels):
        std_values[i] = (sum(sum((image[:, :, i] - mean_values[i]) ** 2)) / (height * width)) ** 0.5
    
    for x in range(channels):
        mean_values[x]=mean_values[x]/255
    for y in range(channels):
        std_values[y]=std_values[y]/255
        
    return mean_values, std_values

# Kích thước ảnh và tham số nhiễu
image_shape = (200, 300, 3)
mean_value = 0
std_dev_value = 5
# original_image=cv2.imread("fig-1.png")
# Tạo ảnh có nhiễu Gauss
noisy_image = generate_noisy_image(image_shape, mean_value, std_dev_value)
gaus_image=Generate_noisy_image_2(image_shape, mean_value, std_dev_value)
Original_image=creatImage(image_shape)
res_image=noisy_image+Original_image
res2_image=gaus_image+Original_image
# Hiển thị ảnh gốc và ảnh có nhiễu
cv2.imshow('Original Image', Original_image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Guas image',gaus_image)
cv2.imshow("result image",res_image)
cv2.imshow("Result 2 image",res2_image)

mean_values, std_values = calculate_mean_std_color(res2_image)
print(f'Mean values: {mean_values}')
print(f'Standard deviation values: {std_values}')

mean_values1, std_values1=calculate_mean_std_function(res2_image)
print(f'Mean values1: {mean_values1}')
print(f'Standard deviation values1: {std_values1}')

# mean_values2, std_values2=calculate_mean_std(res2_image)
# print(f'Mean values2: {mean_values2}')
# print(f'Standard deviation values2: {std_values2}')
cv2.waitKey(0)
cv2.destroyAllWindows()
