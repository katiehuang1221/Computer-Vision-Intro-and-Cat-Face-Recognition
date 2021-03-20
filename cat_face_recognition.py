import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_catface.xml')

people = ['Noodle', 'Chiffon','Cheddar']
# features = np.load('features.npy',allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('cat_face_trained.yml')


# Chiffon
# img = cv.imread(r'/Users/katiehuang/Documents/CV/Cat_faces/Test/IMG_0122.jpeg')
# Cheddar
# img = cv.imread(r'/Users/katiehuang/Documents/CV/Cat_faces/Test/IMG_0123.jpeg')
# Noodle
# img = cv.imread(r'/Users/katiehuang/Documents/CV/Cat_faces/Test/IMG_0124.jpg')
# Chiffon+Cheddar
img = cv.imread(r'/Users/katiehuang/Documents/OpenCV/Cat_faces/Test/IMG_0135.jpg')
# img = cv.imread(r'/Users/katiehuang/Documents/CV/Cat_faces/Test/IMG_0091.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # cv.putText(img, "Hello", (y,x), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0))
    cv.putText(img, str(people[label]), (x,y-50), cv.FONT_HERSHEY_COMPLEX, 2.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    
cv.imshow('Detected Cat Face', img)





cv.waitKey(0)

