import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')                         # Use only for MAC OS
import matplotlib.pyplot as plt

#%matplotlib inline

# Load a color image
img = cv2.imread("../Images/cat.jpg")

# Apply some blurring to reduce noise

# h is the Parameter regulating filter strength for luminance component.
# Bigger h value perfectly removes noise but also removes image details,
# smaller h value preserves details but also preserves some noise

# larger h and hColor values than typical to remove noise at the expense of losing image details

# Experiment with setting h and hColor to a suitable value.

h = 20
hColor = 20

# Default window values
templateWindowSize = 7
searchWindowSize = 21

blur = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)

plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.show()

# Apply a morphological gradient (dilate the image, erode the image, and take the difference

elKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, elKernel)

# openCV's morphologyEx to generate a gradient using the kernel above

plt.imshow(cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))

# Otsu's method - or adjust the level at which thresholding occurs and see what the effect of this is

# Convert gradient to grayscale
gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)

# Generate a matrix called otsu using OpenCV's threshold() function.  Use Otsu's method.

otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Apply a closing operation - using a large kernel here. adjust the size of this kernel and observe the effects
closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, closingKernel)
#plt.show()

# Erode smaller artefacts out of the image - play with iterations to see how it works

# Generate a matrix called eroded using cv2.erode() function over the 'close' matrix.
eroded = cv2.erode(close, None, iterations=6)

plt.imshow(eroded, cmap='gray')
#plt.show()

p = int(img.shape[1] * 0.05)
eroded[:, 0:p] = 0
eroded[:, img.shape[1] - p:] = 0

_, contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the candidates by size, and just keep the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# create two images, initially all zeros (i.e. black)
# One image will be filled with 'Blue' wherever we think there's some starfish
# The other image will be filled with 'Green' whereever we think there's not some starfish
h, w, num_c = img.shape
segmask = np.zeros((h, w, num_c), np.uint8)
stencil = np.zeros((h, w, num_c), np.uint8)

# only one contour, but - in general - expect to have more contours to deal with
for c in contours:
    # Fill in the starfish shape into segmask
    cv2.drawContours(segmask, [c], 0, (255, 0, 0), -1)
    # Lets fill in the starfish shape into stencil as well
    # and then re-arrange the colors using numpy
    cv2.drawContours(stencil, [c], 0, (255, 0, 0), -1)
    stencil[np.where((stencil==[0,0,0]).all(axis=2))] = [0, 255, 0]
    stencil[np.where((stencil==[255,0,0]).all(axis=2))] = [0, 0, 0]

# create a mask image by bitwise ORring segmask and stencil together
mask = cv2.bitwise_or(stencil, segmask)

plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

output = cv2.bitwise_or(mask, img)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.show()