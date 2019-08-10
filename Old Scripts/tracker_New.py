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

resolution = (640, 480) # length x height
framerate = 60
exposure = 1000 # time in milliseconds. 10000 is normal exposure.
CLIENT_IP = '192.168.1.115'
CLIENT_PORT = 8049
fps = FPS()

def main(args):
    try:
        with ThreadedUDPSocket(('', CLIENT_PORT)) as sock:
            stream = VideoStream().start()
            time.sleep(2)
            old_uid = 0
            uid = 0
            key = None
            result = ()
            
            with open('Init_Settings.txt', 'rb') as f:
                result = pickle.load(f)
            
            x_axis_len = result[0]
            y_axis_len = result[1]
            top_l = result[2]
            
            sock.send((CLIENT_IP, CLIENT_PORT),
                          'INIT X:{0:.3f} Y:{1:.3f} S:{2:.3f}'.format(x_axis_len, y_axis_len, args.scale))
            
            start_time = time.time()
            while True:
                while uid == old_uid:
                    frame, uid = stream.read()
                
                #cv2.imshow("Frame", frame)
                #cv2.waitKey(0)
                #cv2.imwrite('Car.jpg', frame)
                
                result = track(frame, top_l, x_axis_len, y_axis_len, args.demo)
                old_uid = uid
                # check if the track function returned nothing
                if not all(result):
                    continue
                
                pos_x = result[0][0]
                pos_y = result[0][1]
                ori = result[1]
                
                #print('DATA X:{0:.3f} Y:{1:.3f} O:{2:.3f}'.format(pos_x, pos_y, ori))
                
                sock.send((CLIENT_IP, CLIENT_PORT),
                          'DATA X:{0:.3f} Y:{1:.3f} O:{2:.3f}'.format(pos_x, pos_y, ori))
                
                print('Results_Time: {:.5f}'.format(time.time() - start_time))
                start_time = time.time()
                

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
        start_time = time.time()
        for uid, frame in enumerate(self.stream):
            
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
            
            #print('Frame time {}'.format(time.time() - start_time))
            start_time = time.time()
            
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

def track(frame, top_l, x_axis_len, y_axis_len, demo):
    fps.start()
    start = time.time()
    im_draw = frame
    # Grayscale, the V channel of HSV is grayscale, just take that.
    _, green, _ = cv2.split(frame)

    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(green, 170, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
    thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
    
    if demo:
        # convert grayscale image back to BGR format for concatenation
        green_formatted = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
        demo_windows1 = np.concatenate((frame, green_formatted), axis=1)

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    erode = cv2.erode(thresh, None, iterations=1)
    #dilate = cv2.dilate(thresh, None, iterations=1)
    #print('Erode_Time: {}'.format(time.time()-start))

    start = time.time()
    if demo:
        thresh_formatted = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        erode_formatted = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)
        demo_windows2 = np.concatenate((thresh_formatted, erode_formatted), axis=1)
    
    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(erode, connectivity=1, background=0, return_num=True)

    # Save points as individual values for maniplulation
    regions = measure.regionprops(labels[0], cache=True)
    
    if demo:
        origin_draw = (int(round(top_l[1])),int(round(top_l[0])))
        boundary = cv2.line(im_draw, origin_draw, (origin_draw[0]+int(round(y_axis_len)), origin_draw[1]), (255,0,0), 1)
        boundary = cv2.line(boundary, origin_draw, (origin_draw[0], origin_draw[1]+int(round(x_axis_len))), (255,0,0), 1)
        boundary = cv2.line(boundary, (origin_draw[0]+int(round(y_axis_len)), origin_draw[1]), (origin_draw[0]+int(round(y_axis_len)), origin_draw[1]+int(round(x_axis_len))), (255,0,0), 1)
        boundary = cv2.line(boundary, (origin_draw[0], origin_draw[1]+int(round(x_axis_len))), (origin_draw[0]+int(round(y_axis_len)), origin_draw[1]+int(round(x_axis_len))), (255,0,0), 1)
        
    if len(regions) != 3:
        print('No points found')
        dir_angle = None
        center_point_rel = None
        direction = frame
    else:
        #print('Points found')
        
        pt1 = regions[0].centroid
        pt2 = regions[1].centroid
        pt3 = regions[2].centroid
        points = [pt1, pt2, pt3]
        center_point_abs = tuple([sum(x)/len(x) for x in zip(*points)])

        # Convert absolute centerpoint to relative centerpoint to top left
        center_point_rel = (center_point_abs[0] - top_l[0], center_point_abs[1] - top_l[1])

        # Find how far along axii center_point is. row is x, col is y
        x = center_point_rel[0]
        y = center_point_rel[1]
        
        # Determine lengths
        pt1_to_pt2 = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        pt1_to_pt3 = math.sqrt((pt3[0] - pt1[0])**2 + (pt3[1] - pt1[1])**2)
        pt2_to_pt3 = math.sqrt((pt3[0] - pt2[0])**2 + (pt3[1] - pt2[1])**2)
        
        # determines point 3 is the front point
        if pt1_to_pt2 <= pt1_to_pt3 and pt1_to_pt2 <= pt2_to_pt3:
            vect1 = (pt3[0] - pt1[0], pt3[1] - pt1[1])
            vect2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
        # determines point 2 is the front point
        elif pt1_to_pt3 <= pt1_to_pt2 and pt1_to_pt3 <= pt2_to_pt3:
            vect1 = (pt2[0] - pt1[0], pt2[1] - pt1[1])
            vect2 = (pt2[0] - pt3[0], pt2[1] - pt3[1])
        # determines point 1 is the front point
        else:
            vect1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
            vect2 = (pt1[0] - pt3[0], pt1[1] - pt3[1])

        # row is x, col is y. Format (row,col)
        dir_vector = (vect1[0] + vect2[0], vect1[1] + vect2[1])
        #print("Direction Vector (array)(row,col)= {point}".format(point = dir_vector))

        # Use vector to get angle with respect to Y(col) axis, angle grows clockwise 0-359
        dir_angle = (math.degrees(math.atan2(dir_vector[0], dir_vector[1])) + 360) % 360
        #print("Direction angle= {point} degrees".format(point = dir_angle))
        if demo:
            center_point_draw = (int(round(center_point_abs[1])), int(round(center_point_abs[0])))
            circle = cv2.circle(boundary, center_point_draw, 2, (0,0,255), -1)

            # make drawable points on axis. row is x, col is y. Format (col,row)
            x_draw = (int(round(top_l[1])), int(round(top_l[0]+x)))
            y_draw = (int(round(top_l[1]+y)), int(round(top_l[0])))
            
            # Draw them row is x, col is y. Format (col,row)
            # top_l along axii
            axii = cv2.line(boundary, origin_draw, x_draw, (0,255,255), 1)
            axii = cv2.line(axii, origin_draw, y_draw, (0,255,255), 1)
            # center_point_abs to axii
            axii = cv2.line(axii, center_point_draw, x_draw, (255,255,0), 1)
            axii = cv2.line(axii, center_point_draw, y_draw, (255,255,0), 1)
            # Circle x and y values
            axii = cv2.circle(axii, x_draw, 2, (255,0,255), -1)
            axii = cv2.circle(axii, y_draw, 2, (255,0,255), -1)
            
            dir_vector_draw = (int(round(dir_vector[1]+center_point_abs[1])), int(round(dir_vector[0]+center_point_abs[0])))
            direction = cv2.line(axii, center_point_draw, dir_vector_draw, (0,255,0), 1)
        
    fps.update()
    fps.stop()
    #print('Calc_Time: {}'.format(time.time()-start))
    #print('FPS: {:.2f}'.format(fps.fps()))
    
    if demo:         
        direction_formatted = cv2.resize(direction, (0, 0), None, 2, 2)
        demo_windows = np.concatenate((demo_windows1, demo_windows2), axis=0)
        demo_windows = np.concatenate((demo_windows, direction_formatted), axis=1)
        demo_windows = cv2.resize(demo_windows, (0, 0), None, 0.5, 0.5)
        cv2.putText(demo_windows, 'FPS: {:.2f}'.format(fps.fps()), (1000, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Demo", demo_windows)
        cv2.waitKey(1) & 0xFF
    
    fps._numFrames = 0
    
    return (center_point_rel, dir_angle)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--demo", type=bool, default=True)
    ap.add_argument("-s", "--scale", type=float, default=0.2753647864651542)
    args = ap.parse_args()
    main(args)
