#import necessary libraries
import cv2
import cvzone
import mediapipe as mp
import os
import time
import numpy as np

################################ FOR VIDEO/WEBCAM FEED ###########################################################

#image paths
pwd = os.getcwd()
img_path = os.path.join(pwd, 'faces.mp4')

#fps reader
fps = cvzone.FPS() #frames per seconds reader

#video capture
cap = cv2.VideoCapture(img_path) #capture video from local path
#cap = cv2.VideoCapture(0) #capture live feed from webcam
cap.set(3, 640) #set witdth of output diplay window
cap.set(4, 480) #set height of output diaply window
cap.set(10, 100) #set brightness of display window

#mediapip face_detection init
faceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
drawing_spec_pts = mpDraw.DrawingSpec((0,255,5),4,2) #set drawing specs for face points
drawing_spec_bb = mpDraw.DrawingSpec((255,255,255),3,5) # set drawing specs for face bounding box
fd = faceDetection.FaceDetection(min_detection_confidence=0.70) #detection confidence level

capture_track = 0

while cap.isOpened():
    suc, img = cap.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = fd.process(rgb_img)
    if faces.detections:
        for id, detection in enumerate(faces.detections):
            print(id, detection)
            mpDraw.draw_detection(img,detection,drawing_spec_pts,drawing_spec_bb)
            ctime = time.ctime()
            capture_track += 1

            cv2.putText(img, f'detect score : {int(detection.score[0]*100)}%',(8,350),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

            #cv2.imwrite(f'captured_images/face_capture{capture_track}.jpg', img) #capture each framewith detected face to "captured_images folder"

    #print(faces)
    fps.update(img, (20,40), (0,255,0),2,2) #display frames per seconds status
    resize = cv2.resize(img, (840, 620), interpolation=cv2.INTER_LINEAR) #resize output image to fit display window

    cv2.imshow('Window', resize) #display image for each frame
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cv2.destroyWindow()
cap.release()

##################################################################################################################

