# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import math
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=r"G:\Python\test7.jpg")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# Convert to HSV mode, mush easier to differentiate colours
green = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Keeping the green
mask = cv2.inRange(green, (30,15,50), (110,255,255))

# Filter keeping green
green[np.where(mask==0)] = 0
cv2.imshow("Image", green)
cv2.waitKey(0)

# Grayscale, the V channel of HSV is grayscale, just take that.
gray = green[:,:,2]
cv2.imshow("Image", gray)
cv2.waitKey(0)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
cv2.imshow("Image", blurred)
cv2.waitKey(0)

thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=1)
cv2.imshow("Image", thresh)
cv2.waitKey(0)

# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
labels = measure.label(thresh, connectivity=1, background=0, return_num=True)
print ("Num of returned labels = {num}".format(num = labels[1]))
mask = np.zeros(thresh.shape, dtype="uint8")

# Save points as individual values for maniplulation
regions = measure.regionprops(labels[0], cache=True)
print ("Num of regions = {num}".format(num = len(regions)))
if len(regions) < 3:
    exit

pt1 = regions[0].centroid
print("Region 1 center = {point}".format(point = pt1))
pt2 = regions[1].centroid
print("Region 2 center = {point}".format(point = pt2))
pt3 = regions[2].centroid
print("Region 3 center = {point}".format(point = pt3))
points = [pt1, pt2, pt3]
center_point = tuple([sum(x)/len(x) for x in zip(*points)])
print("Triangle center = {point}".format(point = center_point))

# Drawing requires integer locations, so round co-ords. Also for some reason flip them. IDK.
# cv2.circle might just be weird? Colour is done in BGR instead of RGB, like it flips all tuples
center_circle = (int(round(center_point[1])), int(round(center_point[0])))
print("Point for circle = {point}".format(point = center_circle))
circle = cv2.circle(image, center_circle, 2, (0,0,255), -1)
cv2.imshow("Image", circle)
cv2.waitKey(0)

# Determine lengths
pt1_to_pt2 = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
pt2_to_pt3 = math.sqrt((pt3[0] - pt2[0])**2 + (pt3[1] - pt2[1])**2)
pt3_to_pt1 = math.sqrt((pt1[0] - pt3[0])**2 + (pt1[1] - pt3[1])**2)

# Find shortest length, opposite corner must be "top" of triangle
if (pt1_to_pt2 < pt2_to_pt3) and (pt1_to_pt2 < pt3_to_pt1):
    # pt3 is "top"
    vect1 = (pt3[0] - pt1[0], pt3[1] - pt1[1])
    vect2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
elif (pt2_to_pt3 < pt1_to_pt2) and (pt2_to_pt3 < pt3_to_pt1):
    # pt1 is "top"
    vect1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    vect2 = (pt1[0] - pt3[0], pt1[1] - pt3[1])
else:
    # pt2 must be "top"
    vect1 = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    vect2 = (pt2[0] - pt3[0], pt2[1] - pt3[1])

dir_vector = (vect1[0] + vect2[0], vect1[1] + vect2[1])
print("Direction Vector = {point}".format(point = dir_vector))
dir_angle = (math.degrees(math.atan2(-dir_vector[0], dir_vector[1])) + 360) % 360
print("Direction Angle = {point} degrees".format(point = dir_angle))

# dest_pt is only needed to draw the line, for actual application only dir_angle will be needed
dest_pt = (center_circle[0] + int(round(dir_vector[1])), center_circle[1] + int(round(dir_vector[0])))
print("Destination point = {point}".format(point = dest_pt))

direction = cv2.line(circle, center_circle, dest_pt, (255,255,255), 1)
cv2.imshow("Image", direction)
cv2.waitKey(0)
