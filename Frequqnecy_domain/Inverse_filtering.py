import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import random as r

def add_gaussian_noise(image, mean, std_dev):
    
    # Tạo nhiễu Gaussian với kích thước và kiểu dữ liệu giống với ảnh đầu vào
    noise = np.random.normal(mean, std_dev, image.shape)
    
    # Thêm nhiễu vào ảnh đầu vào
    noisy_image = image + noise.astype(np.uint8)
    
    # Đảm bảo các giá trị pixel không vượt quá giới hạn 0-255
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image

def addSaltGray(image,n): #add salt-&-pepper noise in grayscale image
    k=0
    salt=True
    ih=image.shape[0]
    iw=image.shape[1]
    noisypixels=(ih*iw*n)/100

    for i in range(ih*iw):
        if k<noisypixels:  #keep track of noise level
                if salt==True:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=255
                        salt=False
                else:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=0
                        salt=True
                k+=1
        else:
            break
    return image

def inverse_filter(Y, H):
    # Biến đổi Fourier cho ảnh đã lọc và kernel của bộ lọc
    Y_fft = np.fft.fft2(Y)
    H_fft = np.fft.fft2(H, Y.shape)
    
    # Tính phổ nghịch đảo của kernel (chú ý tránh chia cho 0)
    H_fft_inv = np.where(H_fft != 0, 1 / H_fft, 0)
    
    # Áp dụng bộ lọc nghịch đảo bằng cách nhân phổ ảnh đã lọc với phổ nghịch đảo của kernel
    I_fft = Y_fft * H_fft_inv
    
    # Thực hiện biến đổi Fourier ngược để lấy lại ảnh gốc
    I_restored = np.fft.ifft2(I_fft).real
    
    return I_restored

H=np.ones((3,3))/9
I = cv2.imread("D:\Python_CV\Frequqnecy_domain\kodim02.png",0)

Y = cv2.filter2D(I, -1, H)
# Y=addSaltGray(Y,0.5)
mean_value=0
std_dev_value=0
Y = add_gaussian_noise(Y,mean_value,std_dev_value)

I2=inverse_filter(Y, H)

f1=np.fft.fft2(I)
f1shift = np.fft.fftshift(f1) 
magnitude_spectrum1 = 20*np.log(np.abs(f1shift))

f2=np.fft.fft2(Y)
f2shift = np.fft.fftshift(f2) 
magnitude_spectrum2 = 20*np.log(np.abs(f2shift))

f3=np.fft.fft2(I2)
f3shift = np.fft.fftshift(f3) 
magnitude_spectrum3 = 20*np.log(np.abs(f3shift))

plt.figure() 
plt.subplot(2, 3, 1) 
plt.imshow(I, cmap="gray"), plt.title("Original image"), plt.axis("off"), 
plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 2) 
plt.imshow(Y, cmap="gray"), plt.title("Image after filter"), plt.axis("off"), 
plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 3) 
plt.imshow(I2, cmap="gray"), plt.title("Image after invertered filter"), 
plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 4) 
plt.imshow(magnitude_spectrum1, cmap="gray"), plt.title("magnitude spectrum original image"), plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 5) 
plt.imshow(magnitude_spectrum2, cmap="gray"), plt.title("magnitude spectrum image after filter"), plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 3, 6) 
plt.imshow(magnitude_spectrum3, cmap="gray"), plt.title("magnitude spectrum invertered filter"), 
plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.show() 

# Bộ lọc càng lớn thì ảnh càng bị mờ thì nó vẫn khôi phục lại I một cách khá tốt
# Nếu có nhiễu với sigma = 0.01 thì nó ra ảnh khôi phục dễ hơn so với ảnh có nhiễu sigma=0.1 

'''Vì nhiễu là nhiễu ở tần số cao thì nếu nó tác động vào vùng tần số cao thì sẽ ít bị ảnh hưởng nhưng nhiễu tác động vào
vùng tần số thấp dẫn đến ảnh hưởng.
I^= Y/H=I+N//H; 
'''
# Nhiễu là ngẫu nhiên nên không thể tính FFT
# Inversering gặp nhiều rủi ro và chưa chắc chắn ( chỉ dùng trong nhiễu rất bé hầu như bằng 0)