import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the martini picture as grayscale.
img = cv2.imread("./Images/martini.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run Canny's edge detection over the martini
edges = cv2.Canny(gray, 100, 200)
plt.grid(False)
plt.axis('off')
plt.imshow(edges, cmap='gray')
plt.show()

kernel = np.ones((10,10),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 6)

plt.imshow(dilation, cmap='gray')
plt.show()

# Find the contours - just external contours to keep post-processing simple
contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the candidates by size, keep the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# Create two related images - one contains the shape of the martini
# glass with black (0's) in the remainder, the other contains
# black where the martini glass is and a colour everywhere else
h, w, num_c = img.shape
segmask = np.zeros((h, w, num_c), np.uint8)
stencil = np.zeros((h, w, num_c), np.uint8)

for c in contours:
    cv2.drawContours(segmask, [c], 0, (255, 0, 0), -1)
    cv2.drawContours(stencil, [c], 0, (255, 0, 0), -1)

# Rearrange the colors in the stencil.  Anything that's black
# replace with green.
stencil[np.where((stencil==[0,0,0]).all(axis=2))] = [0, 255, 0]
# Now its safe to convert the blue to black
stencil[np.where((stencil==[255,0,0]).all(axis=2))] = [0, 0, 0]

# Add the two images (or or them if you prefer)
mask = cv2.bitwise_or(stencil, segmask)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.show()

output = cv2.bitwise_or(mask, img)
plt.imshow(output)
plt.show()