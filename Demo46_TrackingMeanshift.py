#==============================================================================
#   By: Mahnoor Anjum
#   Date: 20/04/2018
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
r,h,c,w = 250,90,400,125 
bounding_window = (c,r,w,h)

lower_color = np.array([40,10,10])
upper_color = np.array([100,255,255])

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

        # apply meanshift 
        ret, bounding_window = cv2.meanShift(dst, bounding_window, term_crit)

        # Draw it on image
        x,y,w,h = bounding_window
        feed = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        feed = cv2.flip(feed,1)
        cv2.imshow('video feed',feed)

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break


cv2.destroyAllWindows()
cap.release()