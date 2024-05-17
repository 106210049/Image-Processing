import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
def kmeans(X, K, max_iters=10): 
    centroids = X[np.random.choice(range(len(X)), K, replace=False)] 
    for i in range(max_iters): 
     # Assign each data point to its closest centroid 
     distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1) 
     labels = np.argmin(distances, axis=-1) 
     # Update the centroids based on the assigned data points 
     for j in range(K): 
      centroids[j] = np.mean(X[labels == j], axis=0) 
    return labels, centroids
# Đọc ảnh màu 
original_img = cv2.imread("D:\Python_CV\Segment and detection\kodim02.png") 
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) 
# Chuyen sang khong gian mau HSV 
HSV_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV) 
# Vector hóa ảnh 
vectorized = HSV_img.reshape((-1, 3)) 
vectorized = np.float32(vectorized) 
# Chạy K-means với K=10
K = 10
labels, centroids = kmeans(vectorized, K) 
# Gán nhãn cho từng pixel và lấy giá trị trung bình của mỗi cluster để tạo ảnh kết quả 
centroids = np.uint8(centroids) 
res = centroids[labels.flatten()] 
result_image = res.reshape((HSV_img.shape)) 
result_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR) 
# Hiển thị ảnh gốc và ảnh phân vùng 
figure_size = 15 
plt.figure(figsize=(figure_size, figure_size)) 
plt.subplot(1, 2, 1), plt.imshow(original_img) 
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(1, 2, 2), plt.imshow(result_image) 
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([]) 
plt.show() 