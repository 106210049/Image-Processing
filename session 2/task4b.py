import numpy as np
import cv2
import random

def creatImage(shape):
    R=70
    height = shape[0]
    width = shape[1]
    center_x=height//2
    center_y=width//2
    image = np.zeros([height, width, 3], dtype=np.uint8)
    
    for x in range(height):
        for y in range(width):
            if (y-center_y)**2+(x-center_x)**2<=R**2:
                image[x][y] = [0, 0 , 255]
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

    return noisy_image

def Generate_noisy_image_2(shape, mean, std_dev):
    gaus_image=np.random.normal(mean,std_dev,(shape[0],shape[1],3))
    return gaus_image

def calculate_mean_std_function(image):
    # Chuyển ảnh về kiểu dữ liệu số thực để tính toán
    image_float = image.astype(np.float32) / 255.0

    # Tính mean và std cho từng kênh màu
    mean_values = np.mean(image_float, axis=(0, 1))
    std_values = np.std(image_float, axis=(0, 1))

    return mean_values, std_values

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
    '''
    Chuyển đổi giá trị pixel của ảnh về kiểu dữ liệu số thực và chia cho 255 nhằm đảm bảo rằng giá trị của pixel nằm trong khoảng [0, 1]. 
    Điều này là quan trọng để làm cho giá trị của pixel thể hiện một tỉ lệ tuyến tính với độ sáng thực sự của điểm ảnh.
    Trong ảnh màu RGB, giá trị pixel thường được biểu diễn trong khoảng từ 0 đến 255. 
    Việc chia giá trị pixel cho 255 giúp chuẩn hóa giá trị về khoảng [0, 1]. 
    Khi các giá trị nằm trong khoảng này, chúng trở thành các số thực trong đoạn [0.0, 1.0], nơi 0.0 thường thể hiện màu đen và 1.0 thể hiện màu trắng.
    Quá trình chuẩn hóa này giúp đồng nhất giá trị của pixel giữa các ảnh và giữ cho các phép toán tính toán được thực hiện trên chúng mang tính tương đối và ổn định hơn. 
    Nó cũng hữu ích khi làm việc với các mô hình học máy hoặc thuật toán xử lý ảnh, vì nó giúp giảm hiện tượng số học và tăng tính ổn định của quá trình học.
    Lưu ý rằng việc này chỉ đúng khi giá trị pixel ban đầu nằm trong khoảng [0, 255]. 
    Trong trường hợp giá trị pixel nằm trong một khoảng khác, bạn cần thực hiện các phép biến đổi tương tự để đảm bảo giá trị pixel thuộc vào đoạn [0, 1].
    '''
    for x in range(channels):
        mean_values[x]=mean_values[x]/255
    for y in range(channels):
        std_values[y]=std_values[y]/255
        
    return mean_values, std_values

def swap_patches(image, patch_size, position1, position2):
    # Sao chép ảnh gốc để tránh ảnh gốc bị thay đổi
    result_image = np.copy(image)

    # Tạo các biến dễ đọc
    row1, col1 = position1
    row2, col2 = position2
    patch_height, patch_width = patch_size

    # Sao chép vùng patch thứ nhất
    patch1 = np.copy(image[row1:row1+patch_height, col1:col1+patch_width])

    # Hoán đổi vùng patch thứ nhất và thứ hai
    result_image[row1-patch_height:row1+patch_height, col1-patch_width:col1+patch_width] = \
    image[row2-patch_height:row2+patch_height, col2-patch_width:col2+patch_width]
    result_image[row2-patch_height:row2+patch_height, col2-patch_width:col2+patch_width] = patch1

    return result_image

def padding(img, radius):
    sz = img.shape
    if len(sz) == 3:
        padding_up = np.zeros([radius, sz[1] + 2*radius, sz[2]], dtype=np.uint8)
        padding_down = np.zeros_like(padding_up, dtype=np.uint8)
        padding_left = np.zeros([sz[0], radius, sz[2]], dtype=np.uint8)
        padding_right = np.zeros_like(padding_left, dtype=np.uint8)
        image = np.vstack([padding_up, np.hstack([padding_left, img, padding_right]), padding_down])
    elif len(sz) == 2:
        padding_up = np.zeros([radius, sz[1] + 2*radius], dtype=np.uint8)
        padding_down = np.zeros_like(padding_up, dtype=np.uint8)
        padding_left = np.zeros([sz[0], radius], dtype=np.uint8)
        padding_right = np.zeros_like(padding_left, dtype=np.uint8)
        image = np.vstack([padding_up, np.hstack([padding_left, img, padding_right]), padding_down])
    return image

def Swap_Padding(img,radius):
    sz = img.shape
    radius = 10
    px = int(np.floor(np.random.rand() * sz[0]))
    py = int(np.floor(np.random.rand() * sz[1]))

    qx = int(np.floor(np.random.rand() * sz[0]))
    qy = int(np.floor(np.random.rand() * sz[1]))

    img = padding(img, radius)

    for i in range(1, sz[0] + 2 * radius):
        for j in range(1, sz[1] + 2 * radius):
            if (i - px - radius)**2 + (j - py - radius)**2 <= radius**2:
                a = img[i, j, :].copy()
                img[i, j, :] = img[i + qx - px, j + qy - py, :]
                img[i + qx - px, j + qy - py, :] = a

    img = img[radius + 1: radius + sz[0], radius + 1: sz[1], :]
    return img

# Kích thước ảnh và tham số nhiễu
image_shape = (300, 400, 3)
mean_value = 0
std_dev_value = 5
height=image_shape[0]
width=image_shape[1]

patch_size = (60, 60)
i=random.randint(0,height)
j=random.randint(0,width)
position1 = (i, j)
position2 = (height-i, width-j)

# Original_image=cv2.imread("fig-1.png")
# Tạo ảnh có nhiễu Gauss
Gaus_noisy_image = Generate_noisy_image_2(image_shape, mean_value, std_dev_value)
Original_image=creatImage(image_shape)
res_image=Gaus_noisy_image+Original_image
result_images = swap_patches(res_image, patch_size, position1, position2)

# Hiển thị ảnh gốc và ảnh có nhiễu
cv2.imshow('Original Image', Original_image)
cv2.imshow('Noisy Image', Gaus_noisy_image)
cv2.imshow("result image",res_image)
cv2.imshow("Final result image",result_images)
cv2.imwrite('result_image_circular.jpg', result_images)

mean_values, std_values = calculate_mean_std_function(result_images)
print(f'Mean values: {mean_values}')
print(f'Standard deviation values: {std_values}')
cv2.waitKey(0)
cv2.destroyAllWindows()
