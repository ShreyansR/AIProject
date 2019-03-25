import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./Images/mrbean.jpg")
b,g,r = cv2.split(img)
org_img = cv2.merge([r, g, b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Remove Noise
kern = np.ones((2, 2), np.uint8)
closeUp = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern, iterations = 2)

# sure background area
sure_bg = cv2.dilate(closeUp, kern, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
boundry = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[boundry == 255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

#Input Image
plt.imshow(org_img)
plt.show()

#Otsu's Image
plt.imshow(thresh, 'gray')
plt.show()

#Close Up Image
plt.imshow(closeUp, 'gray')
plt.show()

#Dilated Image
plt.imshow(sure_bg, 'gray')
plt.show()

#Distance Transformation Image
plt.imshow(dist_transform, 'gray')
plt.show()

#Thresholding Image
plt.imshow(sure_fg, 'gray')
plt.show()

#Boundry Image
plt.imshow(boundry, 'gray')
plt.show()

#Result Image
plt.imshow(img, 'gray')
plt.show()
