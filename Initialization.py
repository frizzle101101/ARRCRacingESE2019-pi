from picamera import PiCamera
from picamera.array import PiRGBArray
from threading import Thread
from imutils.video import FPS
from imutils import contours
from skimage import measure
from udpsocket import ThreadedUDPSocket
import imutils
import cv2
import time
import numpy as np
import argparse
import math
import pickle
import sys

resolution = (640, 480) # length x height
framerate = 60
exposure = 1000 # time in milliseconds. 10000 is normal exposure.
fps = FPS()

def main(args):
    try:
        stream = VideoStream().start()
        time.sleep(2)
        old_uid = 0
        uid = 0
        key = None
        result = ()
        
        while not all(result) or key != ord(' '):
            while uid == old_uid:
                frame, uid = stream.read()
            result = initialize(frame)
            old_uid = uid
            key = cv2.waitKey(1) & 0xFF
        
        cv2.destroyAllWindows()
        
        x_px_len, y_px_len, top_l = result
        
        with open('Init_Settings.txt', 'wb') as f:
            pickle.dump(result, f)
        sys.exit()

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        stream.stop()
        raise SystemExit(0)
    

class VideoStream:
    def __init__(self, resolution=resolution, framerate=framerate):
        # Initialize the camera
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.shutter_speed = exposure
        self.rawCap = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCap,
                                                     format='bgr',
                                                     use_video_port=True)
        # Initialize threading shared variables
        self.frame = None
        self.stopped = False
        self.mtx = np.loadtxt('ORANGE cameraMatrix.txt', delimiter=',', dtype=np.float32)
        self.dist = np.loadtxt('ORANGE cameraDistortion.txt', delimiter=',', dtype=np.float32)
        
    def start(self): 
        # Starts a new thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        # Keep looping infinitely until the thread is stopped
        for uid, frame in enumerate(self.stream):
            start = time.time()
            # Fetch a frame and clear the stream
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (640, 480), 1, (640, 480))
            # undistort
            mapx,mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx,(640, 480), 5)
            dst = cv2.remap(frame.array, mapx, mapy, cv2.INTER_LINEAR)
            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]

            self.frame = (dst, uid)
            self.rawCap.truncate(0)
            
            # If the user stops the program, close the thread
            if self.stopped == True:
                self.stream.close()
                self.rawCap.close()
                self.camera.close()
                return self
    
    def read(self):
        # Return the frame most recently read
        return self.frame
    
    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        return self


def initialize(frame):
    # Isolate Red data
    im_draw = frame
    
    _, _, red = cv2.split(frame)

    # convert grayscale image back to BGR format for concatenation
    red_formatted = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
    demo_windows1 = np.concatenate((frame, red_formatted), axis=1)
    
    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(red, 170, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
    thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)[1]

    blurred_formatted = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    thresh_formatted = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    demo_windows2 = np.concatenate((blurred_formatted, thresh_formatted), axis=1)
    
    # Possibly unnessesary, pylons are very bright.
    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    #thresh = cv2.erode(thresh, None, iterations=1)
    #thresh = cv2.dilate(thresh, None, iterations=1)

    # perform a connected component analysis on the thresholded image
    labels = measure.label(thresh, connectivity=1, background=0, return_num=True)

    # Save points as individual values for maniplulation
    regions = measure.regionprops(labels[0], cache=True)

    if len(regions) != 2:
        x_axis_len = None
        y_axis_len = None
        top_l = None
        axii = frame
    else:
        top_l = regions[0].centroid
        bot_r = regions[1].centroid

        # Get axis lengths, row is x, col is y
        x_axis_len = bot_r[0] - top_l[0]
        y_axis_len = bot_r[1] - top_l[1]
        #print("X-axis length= {point}".format(point = x_axis_len))
        #print("Y-axis length= {point}".format(point = y_axis_len))

        # Draw axis. Only use for Demo
        axii = cv2.line(im_draw, (int(round(top_l[1])),int(round(top_l[0]))), (int(round(top_l[1])),int(round(top_l[0]+x_axis_len))), (0,255,255), 1)
        axii = cv2.line(axii, (int(round(top_l[1])),int(round(top_l[0]))), (int(round(top_l[1]+y_axis_len)),int(round(top_l[0]))), (0,255,255), 1)

    axii_formatted = cv2.resize(axii, (0, 0), None, 2, 2)
    demo_windows = np.concatenate((demo_windows1, demo_windows2), axis=0)
    demo_windows = np.concatenate((demo_windows, axii_formatted), axis=1)
    demo_windows = cv2.resize(demo_windows, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Boundary", demo_windows)
        
        
    return (x_axis_len, y_axis_len, top_l)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--demo", type=bool, default=True)
    ap.add_argument("-s", "--scale", type=float, default=0.2753647864651542)
    args = ap.parse_args()
    main(args)
