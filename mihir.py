import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0) #initiate the camera

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #face detection 
skip = 0
face_data = []
file_name = input("Enter the name of the person")


while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    skip = skip + 1
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    # cv2.imshow("Frame",frame)
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    faces = sorted(faces,key = lambda f: f[2]*f[3],reverse = True)
    #sorting on the largest face area


    # print(faces)
    face_section = frame
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) ## frame,boundaries of face, color,thickness

        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        if skip%10 ==0:
            face_data.append(face_section)
            print(len(face_data))

    
    cv2.imshow("Frame",frame)
    cv2.imshow("Frame Section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF  ## when q pressed then stops
    if(key_pressed == ord('q')):
        break

face_data = np.asarray(face_data)

face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save('./' + file_name + '.npy',face_data)

print("Successfully Saved")

cap.release()
cv2.destroyAllWindows()