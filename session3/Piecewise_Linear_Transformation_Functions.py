import cv2 
import numpy as np 

# Function to map each intensity level to output intensity level. 
def pixelVal(pix, r1, s1, r2, s2): 
	if (0 <= pix and pix <= r1): 
		return (s1 / r1)*pix 
	elif (r1 < pix and pix <= r2): 
		return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
	else: 
		return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 

# Open the image. 
img = cv2.imread("D:\Python_CV\session3\sample.jpg") 
img_gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
# Define parameters. 
# r1 = 70
# s1 = 0
# r2 = 140
# s2 = 255

r1 = 100
s1 = 3
r2 = 160
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array. 
pixelVal_vec = np.vectorize(pixelVal) 

# Apply contrast stretching. 
contrast_stretched = pixelVal_vec(img_gray, r1, s1, r2, s2) 

# Save edited image. 
cv2.imwrite('contrast_stretch.jpg', contrast_stretched) 
cv2.imshow("Original Image",img)
cv2.imshow('contrast_stretch.jpg', contrast_stretched)
cv2.waitKey()
