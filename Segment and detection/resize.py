import cv2
img = cv2.imread('D:\Python_CV\Segment and detection\Pioneer.png')
img=cv2.resize(img,800,200)
cv2.imwrite("Final")