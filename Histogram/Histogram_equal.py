import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_hist(img):
 hist = np.zeros(256)
 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i, j]] += 1
 return hist

def equalize_hist(img):
 # tính tổng pixels
 total_pixels = img.shape[0] * img.shape[1]
 # tính xác suất
 prob = find_hist(img) / total_pixels
 # tính phân bố đồ thị tuyệt đối
 cdf = np.zeros(256)
 for i in range(256):
    cdf[i] = np.sum(prob[:i + 1]) # [0:255]
 # tính phân bố đồ thị tuyệt đối sau khi cân bằng
#  cdf = (cdf * 255).astype(np.uint8)
 max_value=max(cdf)
 min_value=min(cdf)
 cdf=[int(((f-min_value)/(max_value-min_value))*255) for f in cdf]

# thay đổi giá trị pixel
 new_img = np.zeros(img.shape, np.uint8)
 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_img[i, j] = cdf[img[i, j]]
 return new_img

# ãnh xám
# image = cv2.imread("circuit.jpg", 0)
# hist_img = find_hist(image)
# equalize_img = equalize_hist(image)
# hist_equalize_img = find_hist(equalize_img)
# equalize_img_opencv = cv2.equalizeHist(image)
# hist_equalize_img_opencv = find_hist(equalize_img_opencv)
# plt.subplot(3, 2, 1)
# plt.imshow(image, cmap="gray")
# plt.title("Original image"), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 2, 2)
# plt.plot(hist_img)
# plt.title("Histogram of original image"), plt.xlabel("Mức xám"), plt.ylabel("Số lượng")
# plt.subplot(3, 2, 3)
# plt.imshow(equalize_img, cmap="gray")
# plt.title("Equalize image"), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 2, 4)
# plt.plot(hist_equalize_img)
# plt.title("Histogram of equalize image"), plt.xlabel("Mức xám"), plt.ylabel("Số lượng")
# plt.subplot(3, 2, 5)
# plt.imshow(equalize_img_opencv, cmap="gray"), plt.title(
#  "Equalize image opencv"), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 2, 6)
# plt.plot(hist_equalize_img_opencv)
# plt.title("Histogram of equalize image opencv"), plt.xlabel("Mức xám"), plt.ylabel("Số lượng")
# plt.show()


# Ảnh màu
img = cv2.imread("circuit.jpg") 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
hist_img = find_hist(img_YCrCb[:, :, 0]) 
test=img_YCrCb.copy() 
img_YCrCb[:, :, 0] = equalize_hist(img_YCrCb[:, :, 0]) 
img_YCrCb = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR) 
hist_output = find_hist(img_YCrCb[:, :, 0]) 

plt.subplot(2, 2, 1) 
plt.imshow(img), plt.title("Input"), plt.axis("off") 
plt.subplot(2, 2, 2) 
plt.imshow(img_YCrCb), plt.title("Output"), plt.axis("off") 
plt.subplot(2, 2, 3) 
plt.plot(hist_img), plt.title("Hist input"), plt.xlabel( 
"Intensity"), plt.ylabel("Number of pixels") 
plt.subplot(2, 2, 4) 
plt.plot(hist_output), plt.title("Hist output"), plt.xlabel( 
"Intensity"), plt.ylabel("Number of pixels") 
plt.show() 
test[:, :, 0] = cv2.equalizeHist(test[:, :, 0]) 
test = cv2.cvtColor(test, cv2.COLOR_YCrCb2BGR) 
hist_test = find_hist(test[:, :, 0]) 