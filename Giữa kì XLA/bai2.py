import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
input_image = cv2.imread("image.png")
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
(r, c, _) = input_image.shape
mr = np.zeros((r, c))
mg = np.zeros((r, c))
mb = np.zeros((r, c))
mr[::2,::2] = 1
mb[1::2, 1::2] = 1
mg = 1 - mb - mr
I_red = input_image[:, :, 2].astype(np.float64)
I_green = input_image[:, :, 1].astype(np.float64)
I_blue = input_image[:, :, 0].astype(np.float64)
red = mr * I_red
green = mg * I_green
blue = mb * I_blue
demos_picture = np.zeros((r, c, 3), dtype=np.uint8)
demos_picture[:, :, 0] = blue
demos_picture[:, :, 1] = green
demos_picture[:, :, 2] = red
w_R = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/4
w_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])/4
w_B = w_R
blue = cv2.filter2D(blue, -1, w_B)
green = cv2.filter2D(green, -1, w_G)
red = cv2.filter2D(red, -1, w_R)
output_image = np.zeros((r, c, 3), dtype=np.uint8)
output_image[:, :, 0] = blue
output_image[:, :, 1] = green
output_image[:, :, 2] = red

img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
(row, col, rgb) = img.shape
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
multi_img = np.zeros((row, col, 3), dtype=np.uint8)
multi_img[:, :, 0] = blue.astype(np.uint8)
multi_img[:, :, 1] = green.astype(np.uint8)
multi_img[:, :, 2] = red.astype(np.uint8)
F_L = np.array([[-2, 3, -6, 3, -2],
                [3, 4, 2, 4, 3],
                [-6, 2, 48, 2, -6],
                [3, 4, 2, 4, 3],
                [-2, 3, -6, 3, -2]]) * (1/64)
out = red + green + blue
lum = cv2.filter2D(out, -1, F_L)
multi_chr = out - lum
redd = np.zeros((row, col))
greenn = np.zeros((row, col))
bluee = np.zeros((row, col))
redd[::2, ::2] = multi_chr[::2, ::2]
bluee[1::2, 1::2] = multi_chr[1::2, 1::2]
greenn = multi_chr - redd - bluee
smp_chr = np.zeros((row, col, 3), dtype=np.float64)
smp_chr[:, :, 0] = redd
smp_chr[:, :, 1] = greenn
smp_chr[:, :, 2] = bluee
wrb = np.array([[1, 2, 1],
[2, 4, 2],
[1, 2, 1]])/4
wg = np.array([[0, 1, 0],
[1, 4, 1],
[0, 1, 0]])/4
redd = cv2.filter2D(redd, -1, wrb)
greenn = cv2.filter2D(greenn, -1, wg)
bluee = cv2.filter2D(bluee, -1, wrb)
chr = np.zeros((row, col, 3), dtype=np.float64)
chr[:, :, 0] = bluee
chr[:, :, 1] = greenn
chr[:, :, 2] = redd
picture_res = np.zeros((row, col, 3), dtype=np.uint8)
picture_res[:, :, 0] = np.clip(chr[:, :, 0] + lum, 0, 255)
picture_res[:, :, 1] = np.clip(chr[:, :, 1] + lum, 0, 255)
picture_res[:,:, 2] = np.clip(chr[:,:, 2] + lum, 0, 255)


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
psnr1 = calculate_psnr(input_image, output_image)
psnr2 = calculate_psnr(img, picture_res)   
gray_img1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
ssim_value1 = ssim(gray_img1, gray_img2)
gray_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img4 = cv2.cvtColor(picture_res, cv2.COLOR_BGR2GRAY)
ssim_value2 = ssim(gray_img3, gray_img4)
print("PSNR1:", psnr1)
print("SSIM1:", ssim_value1)
print("PSNR2:", psnr2)
print("SSIM2:", ssim_value2)
plt.subplot(2, 2, 1)
plt.imshow(input_image), plt.title('Ảnh gốc phương pháp nội suy tuyến tính'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(output_image), plt.title('Ảnh kết quả phương pháp nội suy tuyến tính'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3)
plt.imshow(img), plt.title('Ảnh gốc phương pháp Alleysson'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(picture_res), plt.title('Ảnh kết quả phương pháp Alleysson'), plt.xticks([]), plt.yticks([])

plt.show()
