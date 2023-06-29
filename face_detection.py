import cv2
import matplotlib.pyplot as plt

childrenImage = cv2.imread("Children-happy.jpg")
model =cv2.CascadeClassifier("haarcascades_haarcascade_frontalface_alt.xml")
#if a proper deep learning method is used the more faces can be establised
faces=model.detectMultiScale(childrenImage,1.3,1)  #if you have 1.3 here this means this will scale by 20-30%
for f in faces:
    print(f)
    x,y,w,h=f
    cv2.rectangle(childrenImage,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(cv2.cvtColor(childrenImage, cv2.COLOR_BGR2RGB))
plt.axis('on')  # Optional: Disable axis labels
plt.show()
print(faces.shape) #[5,4] 5 faces with 4 co-ordinates


# image = cv2.imread("dog.png")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('on')  # Optional: Disable axis labels
# plt.show()
