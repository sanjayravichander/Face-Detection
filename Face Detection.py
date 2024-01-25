# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:29:56 2024

@author: DELL
"""
#Face Detection
import cv2

## Importing a pre-trained model from cv2 library called Harcascade Classifier 
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

## Open the Camera
video_capture = cv2.VideoCapture(0)

## Detecting the Face by drawing a rectangle around the face
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(80, 80))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return 

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    faces = detect_bounding_box(video_frame)  # apply the function we created to the video frame
    cv2.imshow("My Face Detection Project", video_frame)  # display the processed frame in a window named "My Face Detection Project"
    ## To close the cam by pressing the q key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()




