import numpy as np
import cv2
import random as r
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

image=cv2.imread("D:\Python_CV\Filter in space domain\hatgao.jpg",0)
cv2.imshow("Original",image)
salt_image=addSaltGray(image,0.5)
# salt_image=imnoise(image,'salt & pepper',0.05);
cv2.imshow('salt_image',salt_image)
cv2.waitKey(0)