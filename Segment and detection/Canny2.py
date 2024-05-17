import numpy as np
import cv2
def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) # 0-1
    new_img *= 255
    return new_img.astype(np.uint8) # Chuyển đổi về dạng số nguyên 8-bit

def my_canny(img, min_val, max_val, sobel_size=3, is_L2_gradient=False):
    #2. Noise Reduction
    smooth_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    #3. Finding Intensity Gradient of the Image
    Gx = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    Gy = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=sobel_size)
    
    if is_L2_gradient:
        edge_gradient = np.sqrt(Gx*Gx + Gy*Gy)
    else:
        edge_gradient = np.abs(Gx) + np.abs(Gy)
    
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    
    # round angle to 4 directions
    angle = np.abs(angle)
    angle[angle <= 22.5] = 0
    angle[angle >= 157.5] = 0
    angle[(angle > 22.5) * (angle < 67.5)] = 45
    angle[(angle >= 67.5) * (angle <= 112.5)] = 90
    angle[(angle > 112.5) * (angle <= 157.5)] = 135
    
    #4. Non-maximum Suppression
    keep_mask = np.zeros(smooth_img.shape, np.uint8)
    for y in range(1, edge_gradient.shape[0]-1):
        for x in range(1, edge_gradient.shape[1]-1):
            area_grad_intensity = edge_gradient[y-1:y+2, x-1:x+2] # 3x3 area
            area_angle = angle[y-1:y+2, x-1:x+2] # 3x3 area
            current_angle = area_angle[1,1]
            current_grad_intensity = area_grad_intensity[1,1]
            
            if current_grad_intensity > max_val:
                keep_mask[y,x] = 255
            elif current_grad_intensity >= min_val and current_grad_intensity <= max_val:
                if current_angle == 0:
                    if current_grad_intensity > max(area_grad_intensity[1,0], area_grad_intensity[1,2]):
                        keep_mask[y,x] = 255
                elif current_angle == 45:
                    if current_grad_intensity > max(area_grad_intensity[2,0], area_grad_intensity[0,2]):
                        keep_mask[y,x] = 255
                elif current_angle == 90:
                    if current_grad_intensity > max(area_grad_intensity[0,1], area_grad_intensity[2,1]):
                        keep_mask[y,x] = 255
                elif current_angle == 135:
                    if current_grad_intensity > max(area_grad_intensity[0,0], area_grad_intensity[2,2]):
                        keep_mask[y,x] = 255
    
    #5. Hysteresis Thresholding    
    canny_mask = np.zeros(smooth_img.shape, np.uint8)
    canny_mask[keep_mask > 0] = 255

    return scale_to_0_255(canny_mask)

img = cv2.imread('D:\Python_CV\Segment and detection\image.png', 0)
smooth_img = cv2.GaussianBlur(img, (5, 5), 0)
my_canny_output = my_canny(img, min_val=100, max_val=200)
edges=cv2.Canny(smooth_img,100,200)
cv2.imshow("Original image",img)
cv2.imshow('my_canny_output', my_canny_output)
cv2.imshow("edges",edges)
cv2.waitKey(0)
