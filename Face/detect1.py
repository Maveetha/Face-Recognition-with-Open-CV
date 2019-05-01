import cv2
import numpy as np

cam = cv2.VideoCapture(0)

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)

ret, im = cam.read()
locy = int(im.shape[0]/2) # the text location will be in the middle
locx = int(im.shape[1]/2) #           of the frame for this example


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainningData.yml")
id=0

while True:
    ret, im = cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        cv2.putText(im, "Success!", (locx, locy), fontFace, fontScale, fontColorstr(id),(x,y+h),font,255);   
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
