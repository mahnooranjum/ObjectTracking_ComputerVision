#==============================================================================
#   By: Mahnoor Anjum
#   Date: 14/04/2020
#   Codes inspired by:
#   Official Documentation
#==============================================================================
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_green = np.array([45,50,90])
upper_green = np.array([90,255,255])


while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img, lower_green, upper_green)
    
    # Display our object tracker
    frame = cv2.flip(frame, 1)
    mask = cv2.flip(mask, 1)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Original Photo", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()