import cv2 
import math 
import numpy as np 
from PIL import Image
import os
import torch  
def sample(gray,shape,period): 
     
    newImg = np.zeros([shape[0], shape[1], 1], dtype=np.uint8) 
    for i in range(0, shape[0], period): 
        for j in range(0, shape[1], period): 
            sum = 0 
            cnt = 0 
            for x in range(0, period): 
                for y in range(0, period): 
                    if i+x < shape[0] and j+y < shape[1]: 
                        sum += gray[i+x][j+y] 
                        cnt += 1 
            for x in range(0, period): 
                for y in range(0, period): 
                    if i+x < shape[0] and j+y < shape[1]: 
                        newImg[i+x][j+y] = sum // cnt 
    return newImg

image = cv2.imread('D:\Python_CV\Test Open CV\Digital_communication\Input.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
shape=image.shape
period_1 = 4 
period_2 = 2
period_3 = 8
newImg_1=sample(gray,shape,period_1) 
newImg_2=sample(gray,shape,period_2) 
newImg_3=sample(gray,shape,period_3) 


bit_number=[1,2,3,4,5,6,7,8]
BitPerPixels=[0,0,0,0,0,0,0,0]
TotalMemory=[0,0,0,0,0,0,0,0]
rows=image.shape[0]
column=image.shape[1]
bytes=8
for i in range (0,8):
    BitPerPixels[i]=bit_number[i]*3

for j in range (0,8):
    TotalMemory[j]=rows*column*BitPerPixels[j]/bytes

print(f"Dung lượng: {TotalMemory[0]} bytes")
print(f"Dung lượng: {TotalMemory[1]} bytes")
print(f"Dung lượng: {TotalMemory[2]} bytes")
print(f"Dung lượng: {TotalMemory[3]} bytes")
print(f"Dung lượng: {TotalMemory[4]} bytes")
print(f"Dung lượng: {TotalMemory[5]} bytes")
print(f"Dung lượng: {TotalMemory[6]} bytes")
print(f"Dung lượng: {TotalMemory[7]} bytes")

cv2.imshow('Init image', image) 
cv2.imshow('An image with a sampling period of 4', newImg_1) 
cv2.imshow('An image with a sampling period of 2', newImg_2) 
cv2.imshow('An image with a sampling period of 8', newImg_3) 

cv2.waitKey(0) 

