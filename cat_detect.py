import cv2 as cv

img = cv.imread('Photos/cat2.jpg')
cv.imshow('Cat', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

haar_cascade = cv.CascadeClassifier('haar_catface.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of cat faces found = {len(faces_rect)}')


for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)

cv.imshow('Detected Cat Faces', img)



cv.waitKey(0)