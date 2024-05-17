import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

def mean_shift_segmentation(image_path):
    # Đọc ảnh đầu vào
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Chuyển đổi ảnh thành mảng một chiều
    vectorized = original_img.reshape((-1, 3))

    # Ước lượng độ rộng của băng thông
    bandwidth = estimate_bandwidth(vectorized, quantile=0.2, n_samples=500)

    # Áp dụng phân cụm bằng Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(vectorized)

    # Trích xuất nhãn của các cụm và các trung tâm cụm
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    # Tạo ảnh kết quả từ nhãn của các cụm
    segmented_img = cluster_centers[labels].reshape(original_img.shape)

    # Hiển thị ảnh gốc và ảnh phân vùng
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img.astype(np.uint8))
    plt.title('Segmented Image')
    plt.axis('off')

    plt.show()

# Sử dụng chương trình
image_path = "D:\Python_CV\Segment and detection\kodim02.png"  # Đường dẫn đến ảnh của bạn
mean_shift_segmentation(image_path)
