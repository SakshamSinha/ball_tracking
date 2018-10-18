"""
A tracker class for controlling the Tello:

it computes a vector of the ball's direction from the center of the
screen. The axes are shown below (assuming a frame width and height of 600x400):
+y                 (0,200) 
            

Y  (-300, 0)        (0,0)               (300,0)


-Y                 (0,-200)
-X                    X                    +X
 
Based on the tutorial:
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

Usage:
for existing video:
python tracker.py --video ball_tracking_example.mp4
For live feed:
python tracking.py

@author Leonie Buckley and Jonathan Byrne
@copyright 2018 see license file for details
"""

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
    args = vars(ap.parse_args())

    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space. NB the hue range in 
    # opencv is 180, normally it is 360
    # changed green_lower and upper values
    green_lower = (40, 20, 25)
    green_upper = (55,255,120)
    red_lower = (0, 50, 50) 
    red_upper = (20,255,255)
    blue_lower = (110, 50, 50)
    upper_blue = (130,255,255)


    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        vs = VideoStream(src=0).start()

    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])

    # allow the camera or video file to warm up
    time.sleep(2.0)
    stream = args.get("video", False)
    greentracker = Tracker(vs, stream, green_lower, green_upper)

    # keep looping until no more frames
    more_frames = True
    while greentracker.next_frame:
        greentracker.track()
        greentracker.show()
        greentracker.get_frame()

    # if we are not using a video file, stop the camera video stream
    if not args.get("video", False):
        vs.stop()

    # otherwise, release the camera
    else:
        vs.release()

    # close all windows
    cv2.destroyAllWindows()

class Tracker:

    #changed the constructor for integrating with tellotracker.
    def __init__(self, frame, color_lower, color_upper):
        self.color_lower = color_lower
        self.color_upper = color_upper
        self.width = frame.shape[0]
        self.height = frame.shape[1]
        self.midx = int(self.width / 2)
        self.midy = int(self.height / 2)
        self.xoffset = 0
        self.yoffset = 0
        self.frame = frame

    # def __init__(self, vs, stream, color_lower, color_upper):
    #     self.vs = vs
    #     self.stream = stream
    #     self.color_lower = color_lower
    #     self.color_upper = color_upper
    #     self.next_frame = True
    #     self.frame = None
    #     self.get_frame()
    #     height, width, depth = self.frame.shape
    #     self.width = width
    #     self.height = height
    #     self.midx = int(width / 2)
    #     self.midy = int(height / 2)
    #     self.xoffset = 0
    #     self.yoffset = 0

    def get_mids(self):
        return (self.midx,self.midy)

    def get_offsets(self):
        return (self.xoffset, self.yoffset)

    def get_frame(self):
        # grab the current frame
        frame = self.vs.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if self.stream else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            self.next_frame = False
        else:        
            frame = imutils.resize(frame, width=600)
            self.frame = frame

        return frame

    def show(self):
        cv2.putText(self.frame,"Color:", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        cv2.arrowedLine(self.frame, (self.midx, self.midy), 
                                    (self.midx + self.xoffset, self.midy - self.yoffset),
                                    (0,0,255), 5)
        # show the frame to our screen
        cv2.imshow("Frame", self.frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            self.next_frame = False

    def track(self):
        # resize the frame, blur it, and convert it to the HSV
        # color space
        blurred = cv2.GaussianBlur(self.frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(self.frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(self.frame, center, 5, (0, 0, 255), -1)

         
                self.xoffset = int(center[0] - self.midx)
                self.yoffset = int(self.midy - center[1])
            else:
                self.xoffset = 0
                self.yoffset = 0

if __name__ == '__main__':
    main()