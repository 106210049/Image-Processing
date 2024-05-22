import numpy as np
import cv2
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Parameters
a = 0.4
N = 5

# Filter creation
h = np.array([0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2])
ht = h[:, np.newaxis]
W = np.outer(h, h)  # Create a 2D filter from 1D arrays

# Load and convert image to grayscale
I = color.rgb2gray(io.imread('lena.jpg'))

# Initialize the Gaussian pyramid
G = [None] * N
G[0] = I

# Build the Gaussian pyramid
for i in range(1, N):
    temp = convolve(G[i - 1], W, mode='constant')
    h, w = G[i - 1].shape
    G[i] = temp[::2, ::2]

# Initialize the Laplacian pyramid
L = [None] * N

# Build the Laplacian pyramid
for i in range(N):
    if i == N - 1:
        L[i] = G[i]
    else:
        upsampled = cv2.resize(G[i + 1], (G[i].shape[1], G[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        L[i] = G[i] - upsampled
for i in range(N):
    plt.subplot(1, N, i+1)
    plt.imshow(L[i], cmap='gray')
    plt.title(f'L{i+1}')
    plt.axis('off')

plt.show()
# Reconstruct the image
res = L[N - 1]
for i in range(N - 2, -1, -1):
    res = cv2.resize(res, (L[i].shape[1], L[i].shape[0]), interpolation=cv2.INTER_LINEAR) + L[i]

# Display the original and reconstructed images
cv2.imshow('Original Image', I)
cv2.imshow('Reconstructed Image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Calculate and print PSNR and SSIM
# ssim_value = ssim(I, res)
# psnr_value=calculate_psnr(I, res)
# print(f'ssim= {ssim_value}')
# print(f'psnr= {psnr_value}')