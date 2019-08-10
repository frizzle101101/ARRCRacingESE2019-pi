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
ap.add_argument("-i", "--image", default=r"G:\Python\test5.jpg")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it
image = cv2.imread(args["image"])
_ , green, _ = cv2.split(image)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(green, 170, 255, cv2.THRESH_BINARY)[1]
blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=1)

# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
labels = measure.label(thresh, connectivity=2, background=0, return_num=True)

# Save points as individual values for maniplulation
regions = measure.regionprops(labels[0], cache=True)
pt1 = regions[0].centroid
pt2 = regions[1].centroid
pt3 = regions[2].centroid
points = [pt1, pt2, pt3]
center_point = tuple([sum(x)/len(x) for x in zip(*points)])

# Determine lengths
pt1_to_pt2 = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
pt1_to_pt3 = math.sqrt((pt3[0] - pt1[0])**2 + (pt3[1] - pt1[1])**2)

# Check to see if pt1 is NOT the "top" of the triangle (within some tolerance)
if (pt1_to_pt2 <= (pt1_to_pt3 - 5)) or (pt1_to_pt2 >= (pt1_to_pt3 +5)):
    # Check to see if pt2 is the "top" of the triangle
    if pt1_to_pt2 > pt1_to_pt3:
        vect1 = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        vect2 = (pt2[0] - pt3[0], pt2[1] - pt3[1])
        
    # pt3 must be the "top"
    else:
        vect1 = (pt3[0] - pt1[0], pt3[1] - pt1[1])
        vect2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])

# pt1 must be the "top"
else:
    vect1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    vect2 = (pt1[0] - pt3[0], pt1[1] - pt3[1])

dir_vector = (vect1[0] + vect2[0], vect1[1] + vect2[1])
dir_angle = (math.degrees(math.atan2(-dir_vector[0], dir_vector[1])) + 360) % 360
#print("Direction Angle = {point} degrees".format(point = dir_angle))
