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
ap.add_argument("-i", "--image", default=r"G:\Python\test9.jpg")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
im_draw = image
cv2.imshow("Image", image)
cv2.waitKey(0)

# Convert to HSV mode, mush easier to differentiate colours
red = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Keeping red sucks and you need to combine 2 masks
mask0 = cv2.inRange(red, (0,50,50), (20,255,255))
mask1 = cv2.inRange(red, (160,50,50), (180,255,255))
mask = mask0+mask1

# Filter keeping red
red[np.where(mask==0)] = 0
cv2.imshow("Image", red)
cv2.waitKey(0)

# Grayscale, the V channel of HSV is grayscale, just take that.
gray = red[:,:,2]
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
    sys.exit()

pt1 = regions[0].centroid
print("Region 1 center (array)(y,x)= {point}".format(point = pt1))
pt2 = regions[1].centroid
print("Region 2 center (array)(y,x)= {point}".format(point = pt2))
pt3 = regions[2].centroid
print("Region 3 center (array)(y,x)= {point}".format(point = pt3))
pt4 = regions[3].centroid
print("Region 4 center (array)(y,x)= {point}".format(point = pt4))
points = [pt1, pt2, pt3, pt4]

# Use quadrants to determine what point is what
for num in points:
    if 0 <= num[0] <= 239:
        # It is located in the upper half
        if 0 <= num[1] <= 319:
            # It is located in the left half
            top_l = num
        else:
            # It is located in the right half
            top_r = num
    else:
        # It is located in the bottom half
        if 0 <= num[1] <= 319:
            # It is located in the left half
            bot_l = num
        else:
            # It is located in the right half
            bot_r = num


print("Top Left (array)(y,x)= {point}".format(point = top_l))
print("Top Right (array)(y,x)= {point}".format(point = top_r))
print("Bottom Left (array)(y,x)= {point}".format(point = bot_l))
print("Bottom Right (array)(y,x)= {point}".format(point = bot_r))

# Get boundary vectors
vect_x1 = (bot_r[0] - bot_l[0], bot_r[1] - bot_l[1])
vect_x2 = (top_r[0] - top_l[0], top_r[1] - top_l[1])
vect_y1 = (top_l[0] - bot_l[0], top_l[1] - bot_l[1])
vect_y2 = (top_r[0] - bot_r[0], top_r[1] - bot_r[1])

# Draw vectors
boxed = cv2.line(im_draw, (int(round(bot_l[1])),int(round(bot_l[0]))), (int(round(bot_r[1])),int(round(bot_r[0]))), (255,255,255), 1)
boxed = cv2.line(boxed, (int(round(top_l[1])),int(round(top_l[0]))), (int(round(top_r[1])),int(round(top_r[0]))), (255,255,255), 1)
boxed = cv2.line(boxed, (int(round(bot_l[1])),int(round(bot_l[0]))), (int(round(top_l[1])),int(round(top_l[0]))), (255,255,255), 1)
boxed = cv2.line(boxed, (int(round(bot_r[1])),int(round(bot_r[0]))), (int(round(top_r[1])),int(round(top_r[0]))), (255,255,255), 1)
cv2.imshow("Image", boxed)
cv2.waitKey(0)

# Determine lengths (because we already have the vectors we don't have to do as much math)
len_x1 = math.sqrt((vect_x1[0])**2 + (vect_x1[1])**2)
len_x2 = math.sqrt((vect_x2[0])**2 + (vect_x2[1])**2)
len_y1 = math.sqrt((vect_y1[0])**2 + (vect_y1[1])**2)
len_y2 = math.sqrt((vect_y2[0])**2 + (vect_y2[1])**2)

# Determine Angles
angle_x1 = (math.degrees(math.atan2(-vect_x1[0], vect_x1[1])) + 360) % 360
angle_x2 = (math.degrees(math.atan2(-vect_x2[0], vect_x2[1])) + 360) % 360
angle_y1 = (math.degrees(math.atan2(-vect_y1[0], vect_y1[1])) + 360) % 360
angle_y2 = (math.degrees(math.atan2(-vect_y2[0], vect_y2[1])) + 360) % 360

print("X1 length = {point}".format(point = len_x1))
print("X2 length = {point}".format(point = len_x2))
print("X1 angle = {point}".format(point = angle_x1))
print("X2 angle = {point}".format(point = angle_x2))
print("Y1 length = {point}".format(point = len_y1))
print("Y2 length = {point}".format(point = len_y2))
print("Y1 angle = {point}".format(point = angle_y1))
print("Y2 angle = {point}".format(point = angle_y2))

# Calculate Axis vector components
x_axis_len = (len_x1 + len_x2) / 2
y_axis_len = (len_y1 + len_y2) / 2
y_axis_angle =(angle_y1 + angle_y2) / 2
x_axis_angle = y_axis_angle - 90

# Generate Axis vectors
x_axis = (x_axis_len*math.cos(math.radians(x_axis_angle)), x_axis_len*math.sin(math.radians(x_axis_angle)))
y_axis = (y_axis_len*math.cos(math.radians(y_axis_angle)), y_axis_len*math.sin(math.radians(y_axis_angle)))
print("X axis (Cartesian)(x,y)= {point}".format(point = x_axis))
print("Y axis (Cartesian)(x,y)= {point}".format(point = y_axis))

# Draw Axis
axii = cv2.line(im_draw, (int(round(bot_l[1])),int(round(bot_l[0]))), (int(round(bot_l[1]))+int(round(x_axis[0])),int(round(bot_l[0]))-int(round(x_axis[1]))), (0,255,255), 1)
axii = cv2.line(axii, (int(round(bot_l[1])),int(round(bot_l[0]))), (int(round(bot_l[1]))+int(round(y_axis[0])),int(round(bot_l[0]))-int(round(y_axis[1]))), (0,255,255), 1)
cv2.imshow("Image", axii)
cv2.waitKey(0)

###### End of "initialization" ######
# Pass bot_l, x_axis, x_axis_len, y_axis, y_axis_len, y_axis_angle to looping track function
######################################################
# From this point on is a recreation of angle_demo, I'll comment on the new stuff that will be added to the track function

# load the image
image = cv2.imread(args["image"])
im_draw = image
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
print("Region 1 center (array)(y,x)= {point}".format(point = pt1))
pt2 = regions[1].centroid
print("Region 2 center (array)(y,x)= {point}".format(point = pt2))
pt3 = regions[2].centroid
print("Region 3 center (array)(y,x)= {point}".format(point = pt3))
points = [pt1, pt2, pt3]
center_point_abs = tuple([sum(x)/len(x) for x in zip(*points)])
print("Triangle center (array)(y,x)= {point}".format(point = center_point_abs))

##############################################################
############### NEW STUFF TO BE ADDED TO TRACK ###############
##############################################################

from scipy.spatial import distance

# Convert absolute centerpoint to relative centerpoint to bot_l
center_point_rel = (center_point_abs[0] - bot_l[0], center_point_abs[1] - bot_l[1])
center_point_rel_math = (center_point_rel[1], -center_point_rel[0])

# Find how far along X-axis center_point is
x = np.dot(center_point_rel_math, x_axis)/x_axis_len

# Find how far along Y-axis center_point is
y = np.dot(center_point_rel_math, y_axis)/y_axis_len

print("Relative center (array)(y,x)= {point}".format(point = center_point_rel))
print("X (Cartesian)= {point}".format(point = x))
print("Y (Cartesian)= {point}".format(point = y))

# Drawing requires integer locations, so round co-ords. Also for some reason flip them. IDK.
# cv2.circle might just be weird? Colour is done in BGR instead of RGB, like it flips all tuples
center_circle = (int(round(center_point_abs[1])), int(round(center_point_abs[0])))
circle = cv2.circle(image, center_circle, 2, (0,0,255), -1)
cv2.imshow("Image", circle)
cv2.waitKey(0)

# Extra stuff to draw the x position and Y position (don't bother putting this in the function or you'll have to pass it the axii angles) 
x_draw = (x*math.cos(math.radians(x_axis_angle)), x*math.sin(math.radians(x_axis_angle)))
y_draw = (y*math.cos(math.radians(y_axis_angle)), y*math.sin(math.radians(y_axis_angle)))
x_pt = (int(round(x_draw[0]+bot_l[1])), int(round(-x_draw[1]+bot_l[0])))
y_pt = (int(round(y_draw[0]+bot_l[1])), int(round(-y_draw[1]+bot_l[0])))

# Draw them
axii = cv2.line(circle, (int(round(bot_l[1])),int(round(bot_l[0]))), (int(round(bot_l[1]))+int(round(x_draw[0])),int(round(bot_l[0]))-int(round(x_draw[1]))), (0,255,255), 1)
axii = cv2.line(axii, (int(round(bot_l[1])),int(round(bot_l[0]))), (int(round(bot_l[1]))+int(round(y_draw[0])),int(round(bot_l[0]))-int(round(y_draw[1]))), (0,255,255), 1)
axii = cv2.line(axii, x_pt, (int(round(center_point_abs[1])),int(round(center_point_abs[0]))), (255,255,0), 1)
axii = cv2.line(axii, y_pt, (int(round(center_point_abs[1])),int(round(center_point_abs[0]))), (255,255,0), 1)
axii = cv2.circle(axii, x_pt, 2, (255,0,255), -1)
axii = cv2.circle(axii, y_pt, 2, (255,0,255), -1)
cv2.imshow("Image", axii)
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

dir_vector = (vect1[1] + vect2[1], -(vect1[0] + vect2[0]))
dir_vector_draw = (vect1[1] + vect2[1], vect1[0] + vect2[0])
print("Direction Vector (Cartesian)(x,y)= {point}".format(point = dir_vector))
dir_angle_abs = (math.degrees(math.atan2(dir_vector[1], dir_vector[0])) + 360) % 360
dir_angle_rel = (dir_angle_abs + (y_axis_angle - 90) +360) % 360
print("Direction Angle absoulute= {point} degrees".format(point = dir_angle_abs))
print("Direction Angle relative= {point} degrees".format(point = dir_angle_rel))

# dest_pt is only needed to draw the line, for actual application only dir_angle will be needed
dest_pt = (center_circle[0] + int(round(dir_vector_draw[0])), center_circle[1] + int(round(dir_vector_draw[1])))
direction = cv2.line(axii, center_circle, dest_pt, (255,255,255), 1)
cv2.imshow("Image", direction)
cv2.waitKey(0)
