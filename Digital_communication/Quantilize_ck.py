import cv2 
import math 
import numpy as np 


# Hàm lượng tử hóa
def quantization(): 
    
    image = cv2.imread('D:\Python_CV\Test Open CV\Digital_communication\picture.jpg') # Đọc ảnh đầu vào
    image=cv2.resize(image,(255,255)) # Resize về kích thước 256x256
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # đưa về ảnh xám
 
    
    Img6 = gray // 64 * 64 # Lượng tử hóa 4 mức
    
    # Hiển thị ảnh xám ban đầu và ảnh sau khi lượng tử hóa 4 mức
    cv2.imshow("Gray image",gray)
    cv2.imshow("Image 2bit", Img6) 
    
    
    cv2.waitKey(0) 
    cv2.imwrite('Anh xam 4 muc.jpg',Img6) # Lưu ảnh sau khi lượng tử hóa 4 mức
    cv2.imwrite('Anh xam ban dau.jpg',gray)
quantization() # Gọi hàm lượng tử hóa



