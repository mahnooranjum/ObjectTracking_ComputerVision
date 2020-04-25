#==============================================================================
#   By: Mahnoor Anjum
#   Date: 8/12/2018
#   Codes inspired by:
#   Official Documentation
#   opencv-python-tutroals.readthedocs.io
#==============================================================================

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 200,90,400,100 
bounding_window = (c,r,w,h)

lower_color = np.array([90,10,10])
upper_color = np.array([130,255,255])


# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, lower_color, upper_color)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, bounding_window = cv2.CamShift(dst, bounding_window, term_crit)

        # Draw the poly
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        feed = cv2.polylines(frame,[pts],True, 255,2)
        feed = cv2.flip(feed, 1)
        cv2.imshow('video feed',feed)

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break


cv2.destroyAllWindows()
cap.release()