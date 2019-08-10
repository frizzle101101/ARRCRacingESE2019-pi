# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import math
import sys
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="test0.jpg")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
im_draw = image
cv2.imshow("Image", image)
cv2.waitKey(0)

# Isolate Red data
_, green, _ = cv2.split(image)
cv2.imshow("Image", green)
cv2.waitKey(0)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(green, 190, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
cv2.imshow("Image", blurred)
cv2.waitKey(0)

thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)

# Possibly unnessesary, pylons are very bright.
# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
#thresh = cv2.erode(thresh, None, iterations=1)
#thresh = cv2.dilate(thresh, None, iterations=1)
#cv2.imshow("Image", thresh)
#cv2.waitKey(0)

# perform a connected component analysis on the thresholded image
labels = measure.label(thresh, connectivity=1, background=0, return_num=True)

# Save points as individual values for maniplulation
regions = measure.regionprops(labels[0], cache=True)
print ("Num of regions = {num}".format(num = len(regions)))
if len(regions) < 2:
    sys.exit()

top_l = regions[0].centroid
print("Top Left (array)(row,col)= {point}".format(point = top_l))
bot_r = regions[1].centroid
print("Bottom Right (array)(row,col)= {point}".format(point = bot_r))

# Get axis lengths, row is x, col is y
x_axis_len = bot_r[0] - top_l[0]
y_axis_len = bot_r[1] - top_l[1]
print("X-axis length= {point}".format(point = x_axis_len))
print("Y-axis length= {point}".format(point = y_axis_len))

# Draw axis. Only use for Demo
axii = cv2.line(im_draw, (int(round(top_l[1])),int(round(top_l[0]))), (int(round(top_l[1])),int(round(top_l[0]+x_axis_len))), (0,255,255), 1)
axii = cv2.line(axii, (int(round(top_l[1])),int(round(top_l[0]))), (int(round(top_l[1]+y_axis_len)),int(round(top_l[0]))), (0,255,255), 1)
cv2.imshow("Image", axii)
cv2.waitKey(0)

###### End of "initialization" ######
# Pass top_l, x_axis_len, y_axis_len, to looping track function
######################################################
# From this point on is an update to track function

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=r"G:\Python\test6.jpg")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
im_draw = image
cv2.imshow("Image", image)
cv2.waitKey(0)

# Isolate Green data
_, green, _ = cv2.split(image)
cv2.imshow("Image", green)
cv2.waitKey(0)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(green, 170, 255, cv2.THRESH_BINARY)[1]
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

# perform a connected component analysis on the thresholded image
labels = measure.label(thresh, connectivity=1, background=0, return_num=True)
print ("Num of returned labels = {num}".format(num = labels[1]))

# Save points as individual values for maniplulation
regions = measure.regionprops(labels[0], cache=True)
print ("Num of regions = {num}".format(num = len(regions)))
if len(regions) < 3:
    sys.exit()

pt1 = regions[0].centroid
print("Region 1 center (array)(row,col)= {point}".format(point = pt1))
pt2 = regions[1].centroid
print("Region 2 center (array)(row,col)= {point}".format(point = pt2))
pt3 = regions[2].centroid
print("Region 3 center (array)(row,col)= {point}".format(point = pt3))
points = [pt1, pt2, pt3]
center_point_abs = tuple([sum(x)/len(x) for x in zip(*points)])
print("Triangle center (array)(row,col)= {point}".format(point = center_point_abs))

# Convert absolute centerpoint to relative centerpoint to top left
center_point_rel = (center_point_abs[0] - top_l[0], center_point_abs[1] - top_l[1])

# Find how far along axii center_point is. row is x, col is y
x = center_point_rel[0]
y = center_point_rel[1]

print("Relative center (array)(row,col)= {point}".format(point = center_point_rel))
print("X co-ordinate = {point}".format(point = x))
print("Y co-ordinate = {point}".format(point = y))

# Only use for Demo
# Draw triangle center point on image
# Drawing requires integer locations, so round co-ords. Takes col,row
center_point_draw = (int(round(center_point_abs[1])), int(round(center_point_abs[0])))
circle = cv2.circle(im_draw, center_point_draw, 2, (0,0,255), -1)
cv2.imshow("Image", circle)
cv2.waitKey(0)

# make drawable points on axis. row is x, col is y. Format (col,row)
x_draw = (int(round(top_l[1])), int(round(top_l[0]+x)))
y_draw = (int(round(top_l[1]+y)), int(round(top_l[0])))
origin_draw = (int(round(top_l[1])),int(round(top_l[0])))

# Draw them row is x, col is y. Format (col,row)
# top_l along axii
axii = cv2.line(circle, origin_draw, x_draw, (0,255,255), 1)
axii = cv2.line(axii, origin_draw, y_draw, (0,255,255), 1)
# center_point_abs to axii
axii = cv2.line(axii, center_point_draw, x_draw, (255,255,0), 1)
axii = cv2.line(axii, center_point_draw, y_draw, (255,255,0), 1)
# Circle x and y values
axii = cv2.circle(axii, x_draw, 2, (255,0,255), -1)
axii = cv2.circle(axii, y_draw, 2, (255,0,255), -1)
cv2.imshow("Image", axii)
cv2.waitKey(0)
# End draw

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

# row is x, col is y. Format (row,col)
dir_vector = (vect1[0] + vect2[0], vect1[1] + vect2[1])
print("Direction Vector (array)(row,col)= {point}".format(point = dir_vector))

# Use vector to get angle with respect to Y(col) axis, angle grows clockwise 0-359
dir_angle = (math.degrees(math.atan2(dir_vector[0], dir_vector[1])) + 360) % 360
print("Direction angle= {point} degrees".format(point = dir_angle))

# dir_vector_draw is only needed to draw the line
# row is x, col is y. Format (col,row)
dir_vector_draw = (int(round(dir_vector[1]+center_point_abs[1])), int(round(dir_vector[0]+center_point_abs[0])))
direction = cv2.line(axii, center_point_draw, dir_vector_draw, (0,255,0), 1)
cv2.imshow("Image", direction)
cv2.waitKey(0)
# End draw

# Done, send x, y, and dir_angle to unity
