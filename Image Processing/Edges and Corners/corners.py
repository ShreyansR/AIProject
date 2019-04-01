import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the martini image as grayscale.
img = cv2.imread("./Images/martini.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.cornerHarris(gray, 2, 3, 0.04)

#declare threshold margin 
threshold = 0.001

# Threshold may vary
img[corners>threshold * corners.max()]=[255, 0, 0]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
