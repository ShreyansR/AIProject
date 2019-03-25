import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import realpath, normpath

test_image = 'Images/Presenters.jpg'

img = cv2.imread(test_image)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 10), dpi=80)
plt.imshow(img)
plt.show()

cv2path = normpath(realpath(cv2.__file__) + '../../../../../share/OpenCV/haarcascades')

face_cascade_xml_path = "../../anaconda2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
eye_cascade_xml_path = "../../anaconda2/share/OpenCV/haarcascades/haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_xml_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml_path)

# Run the face detector cascade against this grayscale image
scale = 1.032
minNeighbors = 2
foundFaces = face_cascade.detectMultiScale(img, scale, minNeighbors)

print ("We found {}".format(len(foundFaces)) + " faces.")
print ("The array of bounding boxes [top_left_x, top_left_y, width, height] for each face are:\n{}".format(foundFaces))

# setup colors and line thickness for drawing bounding boxes
greenColor = (0, 255, 0)
blueColor = (255, 0, 0)
lineThickness = 2

detectionsImg = img
# now process each face found,
for (fx, fy, fw, fh) in foundFaces:
    # draw the bounding box for each face
    cv2.rectangle(detectionsImg, (fx, fy), (fx + fw, fy + fh), blueColor, lineThickness)

    # Run the eye detector cascade on the subset regions of the image (our
    # "regions of interest" (roi)) that were determined to contain faces
    roi = detectionsImg[fy: fy + fh, fx: fx + fw]
    foundEyes = eye_cascade.detectMultiScale(roi, scale, minNeighbors)

    # now, lets draw bounding boxes for the eyes
    for (ex, ey, ew, eh) in foundEyes:
        cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), greenColor, lineThickness)


plt.figure(figsize=(14, 10), dpi=80)
plt.imshow(img)
plt.show()