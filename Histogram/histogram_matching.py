import cv2
import numpy as np

def find_hist(img):
 hist = np.zeros(256)
 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i, j]] += 1
 return hist

def histogram_matching(image, reference):
    # Tính toán histogram của cả hai hình ảnh
    # hist_image, _ = np.histogram(image.flatten(), 256, [0,256])
    # hist_reference, _ = np.histogram(reference.flatten(), 256, [0,256])
    hist_image=find_hist(image)
    hist_reference=find_hist(reference)
    # Tính toán hàm tính cộng tích lũy (CDF) cho cả hai histogram
    cdf_image = hist_image.cumsum()
    cdf_reference = hist_reference.cumsum()

    # Chuẩn hóa CDF
    cdf_image = (cdf_image - cdf_image.min()) * 255 / (cdf_image.max() - cdf_image.min())
    cdf_reference = (cdf_reference - cdf_reference.min()) * 255 / (cdf_reference.max() - cdf_reference.min())

    # Ánh xạ histogram của hình ảnh đích sang histogram của hình ảnh tham chiếu
    matched = np.interp(image.flatten(), np.arange(0,256), cdf_reference).reshape(image.shape)

    return matched.astype(np.uint8)

# Đọc hình ảnh đích và hình ảnh tham chiếu
image = cv2.imread('circuit.jpg', cv2.IMREAD_GRAYSCALE)
reference = cv2.imread('result_image.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng Histogram Matching
matched_image = histogram_matching(image, reference)

# Hiển thị hình ảnh gốc và hình ảnh sau khi áp dụng Histogram Matching
cv2.imshow('Original Image', image)
cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
