#==============================================================================
#   By: Mahnoor Anjum
#   Date: 14/04/2020
#   Codes inspired by:
#   GeeksforGeeks
#   Official Documentation
#       https://docs.opencv.org/3.4/d2/d55/group__bgsegm.html
#==============================================================================

import numpy as np 
import cv2 
  
scale_percent = 60 # percent of original size

# creating object 
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.createBackgroundSubtractorMOG2() 
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg4 = cv2.bgsegm.createBackgroundSubtractorCNT()

# capture frames from a camera  
cap = cv2.VideoCapture(0)
while(1): 
    # read frames 
    ret, img = cap.read()
    # resize image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
    # apply mask for background subtraction 
    fgmask1 = fgbg1.apply(img)
    fgmask2 = fgbg2.apply(img) 
    fgmask3 = fgbg3.apply(img)
    fgmask4 = fgbg4.apply(img) 

      
    cv2.imshow('Original image', img)
    cv2.imshow('MOG', fgmask1) 
    cv2.imshow('MOG2', fgmask2) 
    cv2.imshow('GMG', fgmask3)
    cv2.imshow('CNT', fgmask3)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

  
cap.release()
cv2.destroyAllWindows() 