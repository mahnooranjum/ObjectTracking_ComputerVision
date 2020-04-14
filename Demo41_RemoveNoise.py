#==============================================================================
#   By: Mahnoor Anjum
#   Date: 14/04/2020
#   Codes inspired by:
#   GeeksforGeeks
#   Official Documentation
#==============================================================================
# importing libraries 
import numpy as np 
import cv2 
scale_percent = 60 # percent of original size
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
  
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
  
# capture frames from a camera  
cap = cv2.VideoCapture(0) 
while(1): 
    ret, img = cap.read() 
    # resize image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    fgmask = fgbg.apply(img) 
      
    cv2.imshow('GMG noise', fgmask) 
      
    # apply transformation to remove noise 
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) 
    cv2.imshow('GMG', fgmask) 
      
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release(); 
cv2.destroyAllWindows(); 
