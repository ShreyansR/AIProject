from skimage.measure import label
import numpy as np
import math as m
import cv2
import time

# A node of a 1-way linked list
class Node(object):
    def __init__(self, data, next):
        self.data = data
        self.next = next

"""
Singaly linked list ADS.
Has append(), remove() and traversal()
Used sparingly to hold different resolutions of images
"""
class SingleList(object):
    head = None
    tail = None

    #displays list data
    def show(self):
        print "Showing list data:"
        current_node = self.head
        while current_node is not None:
            print current_node.data, " -> ",
            current_node = current_node.next
        print None

      #adds another node
    def append(self, data):
        node = Node(data, None)
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
        self.tail = node

        #removes a node
    def remove(self, node_value):
        current_node = self.head
        previous_node = None
        while current_node is not None:
            if current_node.data == node_value:
                # if this is the first node (head)
                if previous_node is not None:
                    previous_node.next = current_node.next
                else:
                    self.head = current_node.next

            # needed for the next iteration
            previous_node = current_node
            current_node = current_node.next

     #visiting each node of the singaly linked list
    def traversal(self, n):
        current_node = self.head
        while n>0:
            current_node = self.tail
        return current_node

"""
Median blur that take1 the center pixel of a 3x3 1quare area of a pixel and 
convert1 it to become the average of the 3x3 area.
"""
def MedianBlur2D(img):
    copy = img
    height, width = img.shape
    members = [0, 0]*9
    for y in range(10+1, height-(10+1)):
        for x in range(10+1, width-(10+1)):
            members[0] = img[y - 1, x - 1]
            members[1] = img[y, x - 1]
            members[2] = img[y + 1, x - 1]
            members[3] = img[y - 1, x]
            members[4] = img[y, x]
            members[5] = img[y + 1, x]
            members[6] = img[y - 1, x + 1]
            members[7] = img[y, x + 1]
            members[8] = img[y + 1, x + 1]

            members.sort()
            copy[y, x] = members[4]
    return copy

"""
Finds the regions of the image which appear to have skin. This is done
by getting the log-opponent image and converting these values into a 
hue, saturation and intensity map. 

Input: An image loaded as a 2D array with 3 channels (RGB).

Output: A binary array with the same size as the input image where
the regions with skin are marked with a '1' and zeros everywhere else.
"""
def skinMap(img):
    b, g, r = cv2.split(img)
    height, width, ch = img.shape

    r_zero_resp = np.nanmin(r[10:height - 10, 10: width - 10])
    g_zero_resp = np.nanmin(g[10:height - 10, 10: width - 10])
    b_zero_resp = np.nanmin(b[10:height - 10, 10: width - 10])
    true_zero = np.nanmin([r_zero_resp, g_zero_resp, b_zero_resp])

    r -= true_zero
    g -= true_zero
    b -= true_zero
    lr = np.zeros((height, width))
    lb = np.zeros((height, width))
    lg = np.zeros((height, width))

    for rows in range(height):
        for cols in range(width):

            lr[rows, cols] = 105*m.log10(r[rows, cols] + 1)
            lb[rows, cols] = 105*m.log10(b[rows, cols] + 1)
            lg[rows, cols] = 105*m.log10(g[rows, cols] + 1)

    I = (lr + lb + lg) / 3
    Rg = lr - lg
    By = lb - ((lg + lr) / 2)
    scale = int(round((width + height) / 320))

    if scale == 0:
        scale = 1

    Rg = MedianBlur2D(Rg)
    By = MedianBlur2D(By)
    I_filt = MedianBlur2D(I)
    MAD = I - I_filt
    MAD = abs(MAD)
    MAD = MedianBlur2D(MAD)
    hue = (np.arctan2(Rg, By) * (180/m.pi))
    saturation = np.sqrt(np.power(Rg, 2) + np.power(By, 2))
    map = np.zeros((height, width))

    for rows in range(height):
        for cols in range(width):
            if (MAD[rows, cols] < 4.5) & (120 < hue[rows, cols]) & (hue[rows,cols] < 160 ) & (10 < saturation[rows, cols]) & (saturation[rows, cols] < 60):
                map[rows, cols] = 1
            if (MAD[rows, cols] < 4.5) & (150 < hue[rows, cols]) & (hue[rows, cols] < 180) & (saturation[rows, cols] > 20) & (saturation[rows, cols] < 80):
                map[rows, cols] = 1

    for y in range(height):
        for x in range(width):
            if (map[y, x] == 1) & (110 <= hue[y, x]) & (0 <= saturation[y, x]) & (saturation[y, x] <= 130):
                map[y, x] = 1;
            else:
                map[y, x] = 0;
    return map

"""
Finds individual skin regions and draws a box around them. Could be vastly improved.

Input: 
"""
def findFaces(map, lower, img, upscale):
    grey = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    ret, grey = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY)
    height, width = map.shape


    for rows in range(height):
        for cols in range(width):
            if grey[rows, cols] < 200:
                grey[rows, cols] = 0
            if map[rows, cols] == 0:
                grey[rows, cols] = 0
    # Destroy Holes
    scale = int(round(18 - (height/100)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grey = cv2.dilate(grey, kernel, iterations=scale)
    grey = cv2.erode(grey, kernel, iterations=scale)

    #Destroy More Holes
    grey = cv2.copyMakeBorder(grey, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    height, width = grey.shape


    print grey.shape, 'grey upscaled'
    al, nums = label(grey, 4, 0, True, 1)

    height, width = grey.shape

    while nums > 0:
        maxx = 0
        maxy = 0
        minx = width
        miny = height
        for rows in range(10, height - 10):
            for cols in range(10, width - 10):
                if al[rows, cols] == nums:
                    if maxx < cols:
                        maxx = cols
                    if minx > cols:
                        minx = cols
                    if maxy < rows:
                        maxy = rows
                    if miny > rows:
                        miny = rows
        nums -= 1
        cv2.rectangle(lower, (maxx, maxy), (minx, miny), (0,255,0), thickness=3, lineType=4, shift=0)

    while upscale > 0:
        grey = cv2.pyrUp(grey)
        lower = cv2.pyrUp(lower)
        upscale -= 1

    cv2.imshow('marked', lower)

# Inital settings.
start_time = time.time()
name = 'exercise-5.jpg'
img = cv2.imread(name)
lower_res = cv2.imread(name)
height, width, ch = img.shape
count = 0
sl = SingleList()

# Lowers the resolution of the image if it gets too large.
while height > 800 or width > 800:
    count += 1
    lower_res = cv2.pyrDown(lower_res)
    sl.append(lower_res)
    height, width, ch = lower_res.shape

# Calls the functions on images. 
print 'scaled down to', lower_res.shape
map = skinMap(lower_res)
findFaces(map, lower_res, img, count)

# Displays the original img and calculates runtime.
cv2.imshow('orignal', img)
endtime = time.time() - start_time
print "Finished after: ", endtime, "seconds"
cv2.waitKey(0)
cv2.destroyAllWindows()
