# import the necessary packages
from objecttracker.centroidtracker import CentroidTracker
from objecttracker.trackableobject import TrackableObject
from imutils.video import FPS
from threading import Thread
import smtplib
import ssl
import numpy as np
import imutils
import dlib
import cv2
import datetime
from datetime import timedelta

# Variables that can be set according to programmer need
# Some thresholding values to be set here as well
interval = timedelta(seconds=60)
startTime = datetime.datetime.now()
endTime = startTime + interval
# For Email purpose
# countOfEntered = 0
# countOfExited = 0
skip_frames = 10
confidence_value = 0.4
# See if we can get data of people from diverse phenotypes
# Talk about future work + limitations
video_file_path = "videos/4.mov"
output_file_path = "output/4.avi"
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalPersonsEntered = 0
totalPersonsExited = 0

# User input:
# Ask if they want a box at a specific size
# - Kid in crib for example:
# - Which pixel you want as the center?
# 4 end points is most easiest

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                               "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

# if a video path was not supplied, grab a reference to the webcam
# if not args.get("input", False):
#     print("[INFO] starting video stream...")
#     vs = VideoStream(src=0).start()
#     time.sleep(2.0)

# otherwise, grab a reference to the video file
# else:
print("[INFO] opening video file...")
vs = cv2.VideoCapture(video_file_path)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a Trackable Object
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# start the frames per second throughput estimator
fps = FPS().start()

# Start Time
startTime = datetime.datetime.now()

print('Given that both the width and height of the video are 500')
width = int(input("Please enter the width of the bounding box: "))
height = int(input("Please enter the height of the bounding box: "))


# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1]

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500, height=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    x1 = W // 2 - (width // 2)
    if (x1 < 0):
        x1 = 0
    
    x2 = W // 2 + (width // 2)
    if (x2 > W):
        x2 = W
    
    y1 = H // 2 - (height // 2)
    if (y1 < 0):
        y1 = 0
    
    y2 = H // 2 + (height // 2)
    if (y2 > H):
        y2 = H
    
    # top side of the rectangle
    cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 255), 2)

    # bottom side of the rectangle
    cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 255), 2)
    # left side of the rectangle
    cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 255), 2)
    # right side of the rectangle
    cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 255), 2)

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_file_path, fourcc, 30,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > confidence_value:
                # extract the index of the class label from the
                # detections list
                
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    print("did not see as person")
                    continue
                print("detected as")
                print(CLASSES[idx])

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"
            # print("tracking here")

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    # print("there are objects")
    # print(objects.items())

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        # print("there are tracked objects")

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            directionY = centroid[1] - np.mean(y) # this is y direction
            to.centroids.append(centroid)

            x = [c[0] for c in to.centroids]
            directionX = centroid[0] - np.mean(x) # this is x direction
            # to.centroids.append(centroid) # we should however append only once

            # check to see if the object has been counted or not
            # if not to.counted:
            if not (to.entered):
                print('entering')
                print(objectID)
                
                if directionY < 0: # moving up
                    if (centroid[1] > y1 and centroid[1] < y2 and centroid[0] > x1 and centroid[0] < x2):
                        totalPersonsEntered += 1
                        to.entered = True
                elif directionY > 0: # moving down
                    if (centroid[1] > y1 and centroid[1] < y2 and centroid[0] > x1 and centroid[0] < x2):
                        totalPersonsEntered += 1
                        to.entered = True
                elif directionX > 0 or directionX < 0: # moving right or moving left
                    if (centroid[0] > x1 and centroid[0] < x2 and centroid[1] > y1 and centroid[1] < y2):
                        totalPersonsEntered += 1
                        to.entered = True
                
            if not (to.exited):
                if directionY > 0: # moving down
                    if (centroid[1] > y2 and centroid[0] > x1 and centroid[0] < x2):
                        totalPersonsExited += 1
                        to.exited = True
                elif directionY < 0:
                    if (centroid[1] < y1 and centroid[0] > x1 and centroid[0] < x2):
                        totalPersonsExited += 1
                        to.exited = True
                elif directionX < 0: # moving left
                    if (centroid[0] < x1 and centroid[1] > y1 and centroid[1] < y2):
                        totalPersonsExited += 1
                        to.exited = True
                elif directionX > 0:
                    if (centroid[0] > x2 and centroid[1] > y1 and centroid[1] < y2):
                        totalPersonsExited += 1
                        to.exited = True
     

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Exited Persons Count", totalPersonsExited),
        ("Entered Persons Count", totalPersonsEntered),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("People Counter", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

#  release the video file pointer
# else:
vs.release()

print("Total People Entered:", totalPersonsEntered)
print("Total People Exited:", totalPersonsExited)

# close any open windows
cv2.destroyAllWindows()
