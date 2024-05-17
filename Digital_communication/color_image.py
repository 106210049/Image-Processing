import cv2 
import math 
import numpy as np 
 
def sample(): 
    image = cv2.imread('D:\Python_CV\circuit.jpg') 
    height = image.shape[0] 
    width = image.shape[1] 
    newImg = np.zeros([height, width, 3], dtype=np.uint8) 
    fre = 2 
    for i in range(0, height, fre): 
        for j in range(0, width, fre): 
            for k in range(0, 3): 
                sum = 0 
                cnt = 0 
                for x in range(0, fre): 
                    for y in range(0, fre): 
                        if i+x < height and j+y < width: 
                            sum += image[i+x][j+y][k] 
                            cnt += 1 
                    for x in range(0, fre): 
                        for y in range(0, fre): 
                            if i+x < height and j+y < width: 
                                newImg[i+x][j+y][k] = sum // cnt 

    cv2.imshow('Init image', image) 
    cv2.imshow('Image', newImg) 
    cv2.waitKey(0)

def quantization(): 
    image = cv2.imread('D:\Python_CV\circuit.jpg') 
 
    Img1 = image // 2 * 2 
    Img2 = image // 4 * 4 
    Img3 = image // 8 * 8 
    Img4 = image // 16 * 16 
    Img5 = image // 32 * 32 
    Img6 = image // 64 * 64 
    Img7 = (image // 128) * 255 
 
    cv2.imshow("Image 8bit", image) 
    cv2.imshow("Image 7bit", Img1) 
    cv2.imshow("Image 6bit", Img2) 
    cv2.imshow("Image 5bit", Img3) 
    cv2.imshow("Image 4bit", Img4) 
    cv2.imshow("Image 3bit", Img5) 
    cv2.imshow("Image 2bit", Img6) 
    cv2.imshow("Image 1bit", Img7) 
    cv2.waitKey(0) 
 
sample() 
quantization() 