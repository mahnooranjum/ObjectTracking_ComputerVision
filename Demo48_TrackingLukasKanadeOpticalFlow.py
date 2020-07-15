#==============================================================================
#   By: Mahnoor Anjum
#   Date: 20/04/2018
#   Codes inspired by:
#   Official Documentation
#   opencv-python-tutroals.readthedocs.io
#   docs.opencv.org
#==============================================================================

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
'''
maxCorners:	
    Maximum number of corners to return.
    If there are more corners than are found,
    the strongest of them is returned. maxCorners <= 0
    implies that no limit on the maximum is set and all
    detected corners are returned. 
qualityLevel:
	 Parameter characterizing the minimal 
    accepted quality of image corners. 
minDistance:
	 Minimum possible Euclidean distance 
    between the returned corners. 
blockSize:
	 Size of an average block for computing a derivative
     covariation matrix over each pixel neighborhood. 
'''



feature_params = dict( maxCorners = 2,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

'''
winSize:
   	size of the search window at each pyramid level. 
maxLevel:
	0-based maximal pyramid level number;
    if set to 0, pyramids are not used (single level),
    if set to 1, two levels are used, and so on;
    if pyramids are passed to input then algorithm will
    use as many levels as pyramids have but no more than maxLevel. 
'''

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break


    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()