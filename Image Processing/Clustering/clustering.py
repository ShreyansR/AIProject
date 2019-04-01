#for image processing technique clustering
#importing libraries 
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import *

# Use the blurred starfish image to blur the background noise

img = Image.open("./Images/StarfishBlur.png")

# Store the image's width and height for later use.
imgWidth = img.size[0]
imgHeight = img.size[1]

# Use 7 features to segment. Experiment with different numbers

numFeatures = 7
# Create a data vector, with 7 values, blue, green, red, x, y, rb, rg
# for every pixel in the image

# Initially used 5 features, but when added red-blue and red-green
# the clustering improved.

Z = np.ndarray(shape=(imgWidth * imgHeight, numFeatures), dtype = float)
Z = np.float32(Z)

# Load data vector with the 7 values
for y in tqdm(range(0, imgHeight), ascii=True):
    for x in range(0, imgWidth):
        xy = (x, y)
        rgb = img.getpixel(xy)
        Z[x + y * imgWidth, 0] = rgb[0]           # blue
        Z[x + y * imgWidth, 1] = rgb[1]           # green
        Z[x + y * imgWidth, 2] = rgb[2]           # red
        # Experimentally, reduce the influence of the x,y components by dividing them by 10
        Z[x + y * imgWidth, 3] = x / 10           # x
        Z[x + y * imgWidth, 4] = y / 10           # y
        Z[x + y * imgWidth, 5] = rgb[2] - rgb[0]  # red - blue
        Z[x + y * imgWidth, 6] = rgb[2] - rgb[1]  # red - green

# Copy of initial vector for OpenCV's K-means implementation.
Z2 = Z.copy()

# Divide into two clusters. So, k = 2
K = 2

# Create our cluster centers.
clusterCenters = np.ndarray(shape=(K,numFeatures))

# Initialise each element of both of vectors
# to rand values (each random number being between the max'es & mins of that feature in Z)
maxVals = np.amax(Z)
minVals = np.amin(Z)
for i, _ in enumerate(clusterCenters):
        clusterCenters[i] = np.random.uniform(minVals, maxVals, numFeatures)

# Created cluster Centers and initialized each clusterCenter's vector

# Create a data vector with an integer to represent whatever cluster a pixel belongs to.
# One entry for each pixel - imgWidth * imgHeight's entries.
pixelClusterMembership = np.ndarray(shape=(imgWidth * imgHeight), dtype = int)

iterations = 10

# For each iteration:
for iteration in tqdm(range(iterations), ascii=True):
    # Part 1: Set each pixel to its cluster

    # use numpy to efficiently subtract both cluster
    # center's vectors from all of the vectors representing
    # the pixels in the image.
    distances = Z[:, None, :] - clusterCenters[None, :, :]

    # Square every element in distances
    distances_sq = distances ** 2

    # Get the sums of the squared vectors
    distance_sum = np.sum(distances_sq, axis=2)

    # get the square root of those sums
    distance = np.sqrt(distance_sum)

    # Pick the indexes of the elements with the smaller of
    # the two distances for each point
    pixelClusterMembership = np.argmin(distance, axis=1)

    # Part 2: update each cluster's centroid
    # print('clusterCenters.shape: ', clusterCenters.shape)
    for i in range(K):
        # Create an empty list of pixels in this cluster
        pixelsInCluster = []

        # For each pixel, retrieve it's cluster membership
        for index, item in enumerate(pixelClusterMembership):
            # if it is member of the current cluster of interest
            if item == i:
                # add it's features to the list of pixels in the cluster
                pixelsInCluster.append(Z[index])

        if len(pixelsInCluster) == 0:
            pixelsInCluster.append(Z[0])

        # Now, for each cluster, simply get the mean of each of its 7 features
        pixelsInCluster = np.array(pixelsInCluster)
        clusterCenters[i] = np.mean(pixelsInCluster, axis=0)

# Display an image

# Replace every pixel in the original image with the rgb values from the mean of the cluster that pixel is now in.
outVec = np.ndarray(shape=(imgWidth * imgHeight, 3), dtype=int)
for index, item in enumerate(tqdm(pixelClusterMembership)):
    outVec[index][0] = int(round(clusterCenters[item][2]))
    outVec[index][1] = int(round(clusterCenters[item][1]))
    outVec[index][2] = int(round(clusterCenters[item][0]))

# Save image
img = Image.new("RGB", (imgWidth, imgHeight))

#display image
for y in tqdm(range(imgHeight), ascii=True):
    for x in range(imgWidth):
        img.putpixel((x, y), (
        int(outVec[y * imgWidth + x][0]), int(outVec[y * imgWidth + x][1]), int(outVec[y * imgWidth + x][2])))

# plt.figure(figsize=(14, 10), dpi=80)
plt.imshow(img)
plt.show()

# OpenCV's K-means
criteria = (cv2.TERM_CRITERIA_MAX_ITER, i+1, 0.1)
ret, label, center = cv2.kmeans(Z2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Convert center back into unsigned bytes
center = np.uint8(center)

# reshape the RGB values from our cv2.kmeans results into
# an image.
rgb = center[:,0:3]
res = rgb[label.flatten()]
img = res.reshape((imgHeight,imgWidth, 3))

#show image
plt.imshow(img)
plt.show()
