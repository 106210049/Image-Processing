import cv2
import numpy as np

image = cv2.imread('circuit.jpg')
"""
Apply identity kernel
"""
kernel1 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
# filter2D() function can be used to apply kernel to an image.
# Where ddepth is the desired depth of final image. ddepth is -1 if...
# ... depth is same as original or source image.
identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

# We should get the same image
cv2.imshow('Original', image)
cv2.imshow('Identity', identity)
print(image.shape)
print(identity.shape)
cv2.waitKey()
# cv2.imwrite('identity.jpg', identity)
cv2.destroyAllWindows()