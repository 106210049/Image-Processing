import numpy as np
import cv2 
import math

def rotate_image(image,angle):
    # Get size of image
    height=image.shape[0]
    width=image.shape[1]

    angle_rad=angle*(math.pi)/180
    
    # Tính tâm quay của ảnh
    center_x=height//2
    center_y=width//2
    
    new_width = int(np.abs(width * math.cos(angle_rad)) + np.abs(height * math.sin(angle_rad)))
    new_height = int(np.abs(height * math.cos(angle_rad)) + np.abs(width * math.sin(angle_rad)))

    # Tạo 1 mảng mới có kích thước giống với kích thước ảnh gốc để lưu ảnh xoay khi xoay
    image_Rotated=np.zeros([new_height,new_width,3],dtype=np.uint8)

    # Di chuyển các điểm ảnh: Biến đổi ma trận xoay trong không gian 2D
    for x in range(height):
        for y in range(width):
            new_x=round(center_x+(x-center_x)*math.cos(angle_rad)-(y-center_y)*math.sin(angle_rad))
            new_y=round(center_y+(x-center_x)*math.sin(angle_rad)+(y-center_y)*math.cos(angle_rad))
            for k in range(3):
                if 0<=new_x<new_height and 0<=new_y<new_width:
                    image_Rotated[new_x][new_y][k]=image[x][y][k]
    return image_Rotated

original_image=cv2.imread('fig-1.png')
rotation_angle=60
rotated_image=rotate_image(original_image,rotation_angle)
gray_img=cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Image',original_image)
cv2.imshow('Rotated Image',rotated_image)
cv2.imshow("Rotated and Gray Image",gray_img)
cv2.waitKey()
