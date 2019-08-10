# OpenCV Image Calibration
# Joshua Yonathan
# June 27th, 2019

import cv2
import sys
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser(description='This script returns a tuple of distortion coefficients from an album of checkerboard images')
parser.add_argument('album_dir', type=str, help='The path of the directory containing the checkerboard images')
parser.add_argument('img_type', type=str, help='The file format of the images (jpg, png, bmp, etc.)')
parser.add_argument('num_rows', type=int, help='The number of rows in the checkboard')
parser.add_argument('num_cols', type=int, help='The number of columns in the checkerboard')
parser.add_argument('dimension', type=int, help='The width of a single square on the checkerboard in millimeters')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

if args.verbose:
    print(args)
    
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.dimension, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((args.num_rows*args.num_cols, 3), np.float32)
objp[:,:2] = np.mgrid[0:args.num_cols,0:args.num_rows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Find album and load images
filename    = args.album_dir + "/*." + args.img_type
images      = glob.glob(filename)

if args.verbose:
    print('{} images found\n'.format(len(images)))
    for i, image in enumerate(images, 1):
        print('{}:\t{}\n'.format(i, image))
        
if len(images) < 9:
    print("A minimum of 9 images are required. Cancelling operation...")
    sys.exit()

nPatternFound = 0
imgNotGood = images[1]

for image in images:
    # Read the file and convert in greyscale
    cv_img = cv2.imread(image)
    cv_gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if args.verbose: print("Reading image: {}".format(image))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(cv_gray_img, (args.num_cols, args.num_rows), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # Clean up the corners
        corners2 = cv2.cornerSubPix(cv_gray_img, corners, (11,11), (-1,-1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(cv_img, (args.num_cols, args.num_rows), corners2, ret)

        print("Pattern found! Press ESC to skip or ENTER to accept")
        cv2.imshow('Image', cv_img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27: # 27 is the ESC button
            print("Image Skipped")
            imgNotGood = image
            cv2.destroyWindow('Image')
            continue

        print("Image accepted")
        nPatternFound += 1
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        cv2.destroyWindow('Image')
        
    else:
        imgNotGood = image
        print("Image Skipped")

if (nPatternFound >= 9):
    print("Found {} good images".format(nPatternFound))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cv_gray_img.shape[::-1], None, None)

    # Undistort an image
    #img = cv2.imread(imgNotGood)
    #h,  w = img.shape[:2]
    #print("Image to undistort: ", imgNotGood)
    #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, w, h))

    # undistort
    #mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,(w, h), 5)
    #dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    #print("ROI: ", x, y, w, h)

    #cv2.imwrite(workingFolder + "/calibresult.png",dst)
    #print("Calibrated picture saved as calibresult.png")
    print("Calibration Matrix: ")
    print(mtx)
    print("Camera Disortion: ")
    print(dist)

    print('Saving to files...')
    filename = "cameraMatrix.txt"
    np.savetxt(filename, mtx, delimiter=',')
    filename = "cameraDistortion.txt"
    np.savetxt(filename, dist, delimiter=',')

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error = mean_error / len(objpoints)

    print("total error: ", mean_error / len(objpoints))

else:
    print("In order to calibrate you need at least 9 good pictures... try again")
    sys.exit()
