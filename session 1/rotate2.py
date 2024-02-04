import numpy as np
from PIL import Image

def rotate_image(image, angle):
    # Chuyển ảnh thành mảng NumPy
    img_array = np.array(image)

    # Chuyển đổi góc xoay từ độ sang radian
    angle_rad = np.radians(angle)

    # Tính toán kích thước mới của ảnh sau khi xoay
    new_width = int(np.abs(img_array.shape[1] * np.cos(angle_rad)) + np.abs(img_array.shape[0] * np.sin(angle_rad)))
    new_height = int(np.abs(img_array.shape[0] * np.cos(angle_rad)) + np.abs(img_array.shape[1] * np.sin(angle_rad)))

    # Tạo mảng mới để chứa ảnh sau khi xoay
    rotated_img_array = np.zeros((new_height, new_width, img_array.shape[2]), dtype=np.uint8)

    # Tính toán tâm xoay của ảnh
    center_x, center_y = new_width // 2, new_height // 2

    # Lấy các pixel từ ảnh gốc và đặt vào vị trí mới sau khi xoay
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            new_x = int((x - center_x) * np.cos(angle_rad) - (y - center_y) * np.sin(angle_rad) + center_x)
            new_y = int((x - center_x) * np.sin(angle_rad) + (y - center_y) * np.cos(angle_rad) + center_y)

            if 0 <= new_x < new_width and 0 <= new_y < new_height:
                rotated_img_array[new_y, new_x, :] = img_array[y, x, :]

    # Chuyển mảng NumPy thành ảnh PIL
    rotated_image = Image.fromarray(rotated_img_array)

    return rotated_image

if __name__ == "__main__":
    # Đường dẫn của ảnh đầu vào
    input_path = "fig-1.png"
    
    # Đọc ảnh từ đường dẫn
    original_image = Image.open(input_path)

    # Góc xoay (đơn vị là độ)
    rotation_angle = 90

    # Gọi hàm để xoay ảnh
    rotated_image = rotate_image(original_image, rotation_angle)

    # Hiển thị ảnh sau khi xoay
    rotated_image.show()
