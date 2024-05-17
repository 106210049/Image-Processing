import numpy as np
import cv2
# Read image
img = cv2.imread('image.jpg',cv2.IMREAD_COLOR) # road.png is the filename

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)

# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 68, minLineLength=15, maxLineGap=250)
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
print("Line Detection using Hough Transform")
cv2.imshow('lanes',img)
cv2.waitKey(0)
cv2.destroyAllWindows()