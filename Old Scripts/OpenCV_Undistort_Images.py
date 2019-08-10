import cv2
import numpy as np

mtx = np.loadtxt('cameraMatrix.txt', delimiter=',', dtype=np.float32)
dist = np.loadtxt('cameraDistortion.txt', delimiter=',', dtype=np.float32)

print(mtx)
print(dist)

# Undistort an image
img = cv2.imread("distorted.jpg")
h,  w = img.shape[:2]
print("Image to undistort: distorted.jpg")
print('Width: ', w)
print('Height: ', h)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print(newcameramtx)
print(roi)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,(w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]

cv2.imwrite("calibresult.png", dst)
print("Calibrated picture saved as calibresult.png")
