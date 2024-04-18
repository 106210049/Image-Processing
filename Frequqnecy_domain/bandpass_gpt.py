import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_bandpass_filter(img, D0_low, D0_high):
    img_double = np.double(img)
    row, col = img_double.shape
    img_fft = np.fft.fft2(img_double)
    img_fft_shift = np.fft.fftshift(img_fft)
    u, v = np.meshgrid(np.arange(col), np.arange(row))
    u = u - (1 + col) / 2
    v = v - (1 + row) / 2
    u = u / col
    v = v / row
    D = np.sqrt(u ** 2 + v ** 2)
    H = np.zeros_like(D)
    H[(D >= D0_low) & (D <= D0_high)] = 1
    img_fft_shift_filtered = img_fft_shift * H
    img_fft_filtered = np.fft.ifftshift(img_fft_shift_filtered)
    img_filtered = np.fft.ifft2(img_fft_filtered)
    img_filtered = np.abs(img_filtered)
    return img_filtered

I = cv2.imread("D:\Python_CV\Frequqnecy_domain\kodim20.png", 0)
I_filtered_by_bandpass = gaussian_bandpass_filter(I, 0.1, 0.7)

f1 = np.fft.fft2(I_filtered_by_bandpass)
f1shift = np.fft.fftshift(f1)
magnitude_spectrum1 = 20 * np.log(np.abs(f1shift))

f2 = np.fft.fft2(I)
f2shift = np.fft.fftshift(f2)
magnitude_spectrum2 = 20 * np.log(np.abs(f2shift))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(I, cmap="gray"), plt.title("Original image"), plt.axis("off"),
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(I_filtered_by_bandpass, cmap="gray"), plt.title("Filtered image"),
plt.axis("off"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3)
plt.imshow(magnitude_spectrum2, cmap="gray"), plt.title("Magnitude Spectrum Original Image"),
plt.axis("off"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(magnitude_spectrum1, cmap="gray"), plt.title("Magnitude Spectrum Bandpass Filtered Image"),
plt.axis("off"), plt.xticks([]), plt.yticks([])
plt.show()