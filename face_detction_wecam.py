import cv2
import matplotlib.pyplot as plt


cam= cv2.VideoCapture(0)


model =cv2.CascadeClassifier("haarcascades_haarcascade_frontalface_alt.xml")
#if a proper deep learning method is used the more faces can be establised

while True:
    success, image = cam.read()
    if not success:
        print("Video reading failed")

    faces=model.detectMultiScale(image,1.3,5)  #if you have 1.3 here this means this will scale by 20-30%
    
    for f in faces:
        print(f)
        x,y,w,h=f
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Image Window", image)
    key= cv2.waitKey(1)
    if key== ord('q'):
       break
