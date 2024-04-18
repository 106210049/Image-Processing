import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
def Generate_noisy_image_2(shape, mean, std_dev):
    gaus_image=np.random.normal(mean,std_dev,(shape[0],shape[1],3))
    return gaus_image

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

I = cv2.imread("D:\Python_CV\Frequqnecy_domain\kodim20.png",0) 
# I = make_mosaic(I) 
I_filtered_by_gaussian_1 = gaussian_lowpass_filter(I, 0.2) 
I_filtered_by_gaussian_2 = gaussian_lowpass_filter(I, 0.5) 
I_filtered_by_bandpass=I_filtered_by_gaussian_2-I_filtered_by_gaussian_1

f1=np.fft.fft2(I_filtered_by_bandpass)
f1shift = np.fft.fftshift(f1) 
magnitude_spectrum1 = 20*np.log(np.abs(f1shift))

f2=np.fft.fft2(I)
f2shift = np.fft.fftshift(f2) 
magnitude_spectrum2 = 20*np.log(np.abs(f2shift))

plt.figure() 
plt.subplot(2, 2, 1) 
plt.imshow(I, cmap="gray"), plt.title("Mosaic image"), plt.axis("off"), 
plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 2) 
plt.imshow(I_filtered_by_bandpass, cmap="gray"), plt.title("I filtered by_bandpass filter"), 
plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 3) 
plt.imshow(magnitude_spectrum2, cmap="gray"), plt.title("magnitude spectrum original image"), plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.subplot(2, 2, 4) 
plt.imshow(magnitude_spectrum1, cmap="gray"), plt.title("magnitude spectrum Bandpass image"), 
plt.axis("off"), plt.xticks([]), plt.yticks([]) 
plt.show() 
