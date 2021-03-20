import os
import cv2 as cv
import numpy as np

cats = ['Noodle', 'Chiffon','Cheddar']
DIR = r'/Users/katiehuang/Documents/CV/Cat_faces/'

haar_cascade = cv.CascadeClassifier('haar_catface.xml')

features = []
labels = []

def create_train():
    for cat in cats:
        path = os.path.join(DIR, person)
        print(path)
        label = cats.index(person)

        for img in os.listdir(path):
            if img != '.DS_Store':
                img_path = os.path.join(path,img)
                img_array = cv.imread(img_path)
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

                for (x,y,w,h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)


create_train()
print('Training done---------')

# print(f'Length of the features = {len(features)}')

features = np.array(features,dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('cat_face_trained.yml')
np.save('cat_features.npy', features)
np.save('cat_labels.npy',labels)
