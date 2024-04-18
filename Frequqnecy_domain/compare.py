import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
def ideal_lowpass_filter(img, D0): 
    img_double = np.double(img) 
    row, col = img_double.shape 
    img_fft = np.fft.fft2(img_double) 
    img_fft_shift = np.fft.fftshift(img_fft) 
    u, v = np.meshgrid(np.arange(col), np.arange(row)) 
    u = u-(1+col)/2 
    v = v-(1+row)/2 
    u = u/col 
    v = v/row 
    D = np.sqrt(u**2+v**2) 
    H = np.zeros_like(D) 
    H[D <= D0] = 1 
    img_fft_shift_filtered = img_fft_shift*H 
    img_fft_filtered = np.fft.ifftshift(img_fft_shift_filtered) 
    img_filtered = np.fft.ifft2(img_fft_filtered) 
    img_filtered = np.abs(img_filtered) 
    return img_filtered
def butterworth_lowpass_filter(img, D0, n): 
    img_double = np.double(img) 
    row, col = img_double.shape 
    img_fft = np.fft.fft2(img_double) 
    img_fft_shift = np.fft.fftshift(img_fft) 
    u, v = np.meshgrid(np.arange(col), np.arange(row)) 
    u = u-(1+col)/2 
    v = v-(1+row)/2 
    u = u/col 
    v = v/row 
    D = np.sqrt(u**2+v**2) 
    H = np.zeros_like(D) 
    H = 1/(1+(D/D0)**(2*n)) 
    img_fft_shift_filtered = img_fft_shift*H 
    img_fft_filtered = np.fft.ifftshift(img_fft_shift_filtered) 
    img_filtered = np.fft.ifft2(img_fft_filtered) 
    img_filtered = np.abs(img_filtered) 
    return img_filtered 
def gaussian_lowpass_filter(img, D0): 
    img_double = np.double(img) 
    row, col = img_double.shape 
    img_fft = np.fft.fft2(img_double)
    img_fft_shift = np.fft.fftshift(img_fft) 
    u, v = np.meshgrid(np.arange(col), np.arange(row)) 
    u = u-(1+col)/2 
    v = v-(1+row)/2 
    u = u/col 
    v = v/row 
    D = np.sqrt(u**2+v**2) 
    H = np.zeros_like(D) 
    H = np.exp(-(D**2)/(2*(D0**2))) 
    img_fft_shift_filtered = img_fft_shift*H 
    img_fft_filtered = np.fft.ifftshift(img_fft_shift_filtered) 
    img_filtered = np.fft.ifft2(img_fft_filtered) 
    img_filtered = np.abs(img_filtered) 
    return img_filtered 
def make_mosaic(img): 
    row, col = img.shape[:2] 
    mr = np.zeros((row, col)) 
    mg = np.zeros((row, col)) 
    mb = np.zeros((row, col)) 
    mr[::2, ::2] = 1 
    mb[1::2, 1::2] = 1 
    mg = 1 - mr - mb 
    red = img[:, :, 2] 
    green = img[:, :, 1] 
    blue = img[:, :, 0] 
    red = np.multiply(red.astype(float), mr) 
    green = np.multiply(green.astype(float), mg) 
    blue = np.multiply(blue.astype(float), mb) 
    # multi_img = np.zeros((row, col, 3), dtype=np.uint8)
    # multi_img[:, :, 0] = blue.astype(np.uint8) 
    # multi_img[:, :, 1] = green.astype(np.uint8) 
    # multi_img[:, :, 2] = red.astype(np.uint8) 
    mosaic_image = red+green+blue 
    return mosaic_image 
I = cv2.imread("D:\Python_CV\Frequqnecy_domain\kodim14.png") 
I_mosaic = make_mosaic(I) 
I_filtered_by_ilpf = ideal_lowpass_filter(I_mosaic, 0.1) 
I_filtered_by_butterworth = butterworth_lowpass_filter(I_mosaic, 0.1, 2) 
I_filtered_by_gaussian = gaussian_lowpass_filter(I_mosaic, 0.1) 
plt.figure() 
plt.subplot(2, 2, 1) 
plt.imshow(I_mosaic, cmap="gray"), plt.title("Mosaic image"), plt.axis("off"), 
plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 2) 
plt.imshow(I_filtered_by_ilpf, cmap="gray"), plt.title("Ideal lowpass filter"), 
plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 3) 
plt.imshow(I_filtered_by_butterworth, cmap="gray"), plt.title("Butterworth lowpass filter"), plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 4) 
plt.imshow(I_filtered_by_gaussian, cmap="gray"), plt.title("Gaussian lowpass filter"), 
plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.show() 