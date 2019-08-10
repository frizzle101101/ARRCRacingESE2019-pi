from picamera import PiCamera
from picamera.array import PiRGBArray
from threading import Thread
from imutils.video import FPS
from imutils import contours
from skimage import measure
from udpsocket import ThreadedUDPSocket
import picamera
import imutils
import cv2
import time
import numpy as np
import argparse
import math
import multiprocessing
import queue
import pickle
import os
from multiprocessing import Queue

Frame_Q = Queue(maxsize=1)
Undistorted_Q = Queue(maxsize=1)
Filtered_Q = Queue(maxsize=1)
Results_Q = Queue(maxsize=1)
resolution = (640, 480) # length x height
framerate = 30
exposure = 1000 # time in milliseconds. 10000 is normal exposure.
CLIENT_IP = '192.168.1.115'
CLIENT_PORT = 8049
fps = FPS()
frame_list = []
undistort_list = []

def main(args):
    try:
        with ThreadedUDPSocket(('', CLIENT_PORT)) as sock:
            stream = VideoStream().start()
            time.sleep(2)
            
            with open('Init_Settings.txt', 'rb') as f:
                result = pickle.load(f)
            
            x_axis_len = result[0]
            y_axis_len = result[1]
            top_l = result[2]

            #undistort = multiprocessing.Process(target=undistorter, args=(Frame_Q, Undistorted_Q))
            #undistort.daemon = True
            #undistort.start()
            
            filter_image = multiprocessing.Process(target=filterer, args=(Undistorted_Q, Filtered_Q))
            filter_image.daemon = True
            filter_image.start()
            
            calc = multiprocessing.Process(target=calculator, args=(Filtered_Q, Results_Q, top_l))
            calc.daemon = True
            calc.start()
                
            sock.send((CLIENT_IP, CLIENT_PORT),
                          'INIT X:{0:.3f} Y:{1:.3f} S:{2:.3f}'.format(x_axis_len, y_axis_len, args.scale))
            
            start_time = time.time()
            while True:
                while Results_Q.empty():
                    time.sleep(0.002)
                
                x, y, dir_angle = Results_Q.get()
                sock.send((CLIENT_IP, CLIENT_PORT),
                      'DATA X:{0:.3f} Y:{1:.3f} O:{2:.3f}'.format(x, y, dir_angle))
                
                print('Results_Time: {:.5f}'.format(time.time() - start_time))
                start_time = time.time()
                time.sleep(0.030)

    except KeyboardInterrupt:
        sock.close()
        #undistort.terminate()
        #undistort.join()
        filter_image.terminate()
        filter_image.join()
        calc.terminate()
        calc.join()
        cv2.destroyAllWindows()
        stream.stop()
        raise SystemExit(0)
    

class VideoStream:
    def __init__(self, resolution=resolution, framerate=framerate):
        os.system('sudo modprobe bcm2835-v4l2')
        
        # Initialize the camera
        #self.camera = PiCamera()
        #self.camera.resolution = resolution
        #self.camera.framerate = framerate
        #self.camera.shutter_speed = exposure
        #self.rawCap = PiRGBArray(self.camera, size=resolution)
        #self.stream = self.camera.capture_continuous(self.rawCap,
                                                     #format='bgr',
                                                     #use_video_port=True)
        self.stream = cv2.VideoCapture(0)
        
        
        # Initialize threading shared variables
        self.frame = None
        self.stopped = False
        
    def start(self): 
        # Starts a new thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        
        start_time = time.time()
        
        #try:
        while True:
            ret, self.frame = self.stream.read()

            # Some filtering
            _, green, _ = cv2.split(self.frame)

            try:
                clear_garbage = Frame_Q.get(False)
                Frame_Q.put(green)
            except:
                Frame_Q.put(green)
            
            
            #frame_list.append(time.time() - start_time)
            #if (len(frame_list)) == 1000:
            #    print('AVERAGE FRAME TIME: ', sum(frame_list)/1000)
            #    raise SystemExit(0)
            
            try:
                time.sleep(0.029-(time.time() - start_time))
            except:
                pass
                
            print('Frame time {}'.format(time.time() - start_time))
            
            start_time = time.time()

        #except Exception as err:
            #print(err)
            #self.stream.release()
        # Keep looping infinitely until the thread is stopped
        """
        while True:
            try:
                for uid, frame in enumerate(self.stream):
            
                    data = frame.array
                    
                    try:
                        clear_garbage = Frame_Q.get(False)
                        Frame_Q.put(data)
                    except queue.Empty:
                        if Frame_Q.empty():
                            Frame_Q.put(data)
                    #except queue.Full          
                     #   clear_garbage = Frame_Q.get(True, 0.001)
                      #  Frame_Q.put(data, True, 0.001)
                            
                    self.rawCap.truncate(0)
                    
                    time_list.append(time.time() - start_time)
                    if (len(time_list)) == 1000:
                        print('AVERAGE TIME: ', sum(time_list)/1000)
                        raise SystemExit(0)
                    #print('Frame time {}'.format(time.time() - start_time))
                    start_time = time.time()
                    
                    # If the user stops the program, close the thread
                    if self.stopped == True:
                        self.stream.close()
                        self.rawCap.close()
                        self.camera.close()
                        return self
            except:
                self.stream.close()
                #self.rawCap.close()
                self.camera.close() """
                
    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        return self
    
    
def undistorter(Frame_Q, Undistorted_Q):
    
    mtx = np.loadtxt('ORANGE cameraMatrix.txt', delimiter=',', dtype=np.float32)
    dist = np.loadtxt('ORANGE cameraDistortion.txt', delimiter=',', dtype=np.float32)
    
    while True:
        try:
            #start_time = time.time()
            frame = Frame_Q.get()
            start_time = time.time()
            
            
            
            #print('Undistort time: {}'.format(time.time()-start_time))
            
            
            
            try:
                clear_garbage = Undistorted_Q.get(False)
                Undistorted_Q.put((frame, newcameramtx, roi))
                
            except:
                Undistorted_Q.put((frame, newcameramtx, roi))
                
            #undistort_list.append(time.time() - start_time)
            #if (len(undistort_list)) == 1000:
            #    print('AVERAGE UNDISTORT TIME: ', sum(undistort_list)/1000)
            #    undistort_list = 0
            print('Undistort time: {}'.format(time.time()-start_time))
              
        except queue.Empty:
            time.sleep(0.002)
                

def filterer(Undistorted_Q, Filtered_Q):
    
    mtx = np.loadtxt('ORANGE cameraMatrix.txt', delimiter=',', dtype=np.float32)
    dist = np.loadtxt('ORANGE cameraDistortion.txt', delimiter=',', dtype=np.float32)
    
    while True:
        try:
            #frame, newcameramtx, roi = Undistorted_Q.get()
            frame = Frame_Q.get()
            start_time = time.time()

            # Fetch a frame and clear the stream
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 1, (640, 480))

            # undistort
            mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,(640, 480), 5)
            dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            
            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]           

            thresh = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)[1]
            #dilate = cv2.dilate(thresh, None, iterations=1)

            # Might add the connected component analysis to this part later, will have too see how long it takes as is
            try:
                clear_garbage = Filtered_Q.get(False)
                Filtered_Q.put(thresh)
            except:
                Filtered_Q.put(thresh)
            
            print('Filter time: {}'.format(time.time()-start_time))
        except:
            time.sleep(0.002) 
    

def calculator(Filtered_Q, Results_Q, top_l):
    while True:
        try:            
            thresh = Filtered_Q.get()
            start_time = time.time()

            # threshold the image to reveal light regions in the
            # blurred image
            
            blurred = cv2.GaussianBlur(thresh, (7, 7), 0)

            thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)[1]
            # perform a series of erosions and dilations to remove
            # any small blobs of noise from the thresholded image
            erode = cv2.erode(thresh, None, iterations=1)
            # perform a connected component analysis on the thresholded
            # image, then initialize a mask to store only the "large"
            # components
            labels = measure.label(erode, connectivity=1, background=0, return_num=True)

            # Save points as individual values for maniplulation
            regions = measure.regionprops(labels[0], cache=True)
    
            if len(regions) != 3:
                print('No points found')
            else:     
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
                try:
                    clear_garbage = Results_Q.get(False)
                    Results_Q.put((x,y,dir_angle), False)
                except:
                     Results_Q.put((x,y,dir_angle), False)
                    
                print('Calc_Time: {}'.format(time.time()-start_time))
                #start_time = time.time()
        except:
            time.sleep(0.002)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--demo", type=bool, default=True)
    ap.add_argument("-s", "--scale", type=float, default=0.2753647864651542)
    args = ap.parse_args()
    
    main(args)
