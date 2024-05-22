import cv2
import matplotlib.pyplot as plt
import glob

# Đường dẫn đến thư mục chứa ảnh
image_folder = 'D:\Python_CV\CK XLA\Mask'

# Đọc tất cả các file ảnh trong thư mục
image_files = glob.glob(f'{image_folder}/*.png')  # Bạn có thể thay đổi định dạng ảnh nếu cần

# Đảm bảo rằng chúng ta có đủ ảnh để hiển thị
num_images = 20  # Số lượng ảnh cần hiển thị
if len(image_files) < num_images:
    raise ValueError("Không đủ ảnh trong thư mục để hiển thị.")

# Đọc và lưu trữ các ảnh vào một danh sách
images = []
for i in range(num_images):
    img = cv2.imread(image_files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi màu từ BGR sang RGB
    images.append(img)

# Hiển thị các ảnh trên figure với 2 hàng, mỗi hàng 5 ảnh
fig, axes = plt.subplots(4, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.axis('off')  # Tắt hiển thị trục tọa độ
    ax.set_title(f'Image {i+1}')

plt.tight_layout()
plt.show()
