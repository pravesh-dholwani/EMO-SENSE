import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
from flask import Flask,request, render_template, Response, session, redirect
from functools import wraps
from datetime import datetime
import datetime as dt
import pymongo
import cv2
import tensorflow as tf
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cvlib


camera = cv2.VideoCapture(0)
classifier =load_model('model/Emotion_Detection.h5')
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

path="Customers"

images = []
mylist = os.listdir(path)
print(mylist)
classNames=[]
print(cv2.imread(f'{path}/{mylist[0]}'))
for i in range(len(mylist)):
    curImg = cv2.imread(f'{path}/{mylist[i]}')
    images.append(curImg)
    print(os.path.splitext(mylist[i])[0].split(".")[0])
    classNames.append(os.path.splitext(mylist[i])[0].split(".")[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

# cap  = cv2.VideoCapture(0)
# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#     faces_in_frame = face_recognition.face_locations(imgS)
#     encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
#     for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
#         matches = face_recognition.compare_faces(encoded_face_train, encode_face)
#         faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
#         matchIndex = np.argmin(faceDist)
#         print(matchIndex)
#         print(matches[matchIndex])
#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper().lower()
#             y1,x2,y2,x1 = faceloc
#             # since we scaled down by 4 times
#             y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
#             cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             # markAttendance(name)
#     cv2.imshow('webcam', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
          try:
            #print("hello")
            faces,c=cvlib.detect_face(frame)
  #     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  #     faces = face_classifier.detectMultiScale(gray,1.3,5)

            for face in faces:
              #   print(face)
                (startX,startY) = face[0],face[1]
                (endX,endY) = face[2],face[3]
                # draw rectangle over face
        #         roi=
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                roi_rgb=rgb[startY-30:endY+30,startX-30:endX+30]
                roi_gray = gray[startY-30:endY+30,startX-30:endX+30]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                roi_rgb=cv2.resize(roi_rgb, (0,0), None, 0.25,0.25)
                encoded_face = face_recognition.face_encodings(roi_rgb)[0]


                ##Recognize face
                matches = face_recognition.compare_faces(encoded_face_train, encoded_face)
                faceDist = face_recognition.face_distance(encoded_face_train, encoded_face)
                matchIndex = np.argmin(faceDist)
                print(matchIndex)
                print(matches[matchIndex])
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper().lower()
                    # y1,x2,y2,x1 = faceloc
                    # # since we scaled down by 4 times
                    # y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    #cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(frame,name, (startX,startY-25), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


                if np.sum([roi_gray])!=0:
                  roi = roi_gray.astype('float')/255.0
                  roi = img_to_array(roi)
                  roi = np.expand_dims(roi,axis=0)

              # make a prediction on the ROI, then lookup the class

                  preds = classifier.predict(roi)[0]
                  print("\nprediction = ",preds)
                  label=class_labels[preds.argmax()]
                  maxi=-1
                  if label=="Neutral":
                      for i in range(5):
                          if i==2:
                              continue
                          else:
                              if maxi<preds[i]:
                                  maxi=preds[i]
                                  label=class_labels[i]
                  #print("\nprediction max = ",preds.argmax())
                  # print("\nlabel = ",label)
                  label_position = (startX,startY)
                  cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                else:
                  cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
          except Exception as e:
            print(e)
          cv2.imshow('webcam', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
           break