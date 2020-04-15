#==============================================================================
#   By: Mahnoor Anjum
#   Date: 14/04/2020
#   Codes inspired by:
#   Official Documentation
#   LearnOpenCV.com
#==============================================================================

import cv2
import sys


tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[1]


if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Read frame
cap = cv2.VideoCapture("videos/chaplin.mp4")

# Read first frame
ret, frame = cap.read()
if not ret:
    print ('Cannot read video file')
    sys.exit()


# Uncomment the line below to select a different bounding box
bounder = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, bounder)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    ret, bounder = tracker.update(frame)

    # Draw bounding box
    if ret:
        # Tracking success
        p1 = (int(bounder[0]), int(bounder[1]))
        p2 = (int(bounder[0] + bounder[2]), int(bounder[1] + bounder[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        cv2.putText(frame, "Tracking failure", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),3)

    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break


# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()
