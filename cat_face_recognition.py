import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_catface.xml')

cats = ['Noodle', 'Chiffon','Cheddar']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('cat_face_trained.yml')

# Image for testing
img = cv.imread(r'/Users/katiehuang/Documents/OpenCV/Cat_faces/Test/IMG_0135.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {cats[label]} with a confidence of {confidence}')

    # cv.putText(img, "Hello", (y,x), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0))
    cv.putText(img, str(cats[label]), (x,y-50), cv.FONT_HERSHEY_COMPLEX, 2.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    
cv.imshow('Detected Cat Face', img)
cv.waitKey(0)

