import cv2    
import os

haar_algo_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'
sub_data = 'images'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)

(width, height) = (130, 110)    
face_cascade = cv2.CascadeClassifier(haar_algo_file)
webcam = cv2.VideoCapture(0)

count = 1
while count < 31:
    print(count)
    ret, img = webcam.read()
    if not ret:
        break
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 4)
    if len(faces) > 0:
        message = "Person Detected"
        for(x,y,w,h) in faces:
           cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
           face = grayImg[y:y+h, x:x+w]
           face_resize = cv2.resize(face,(width,height))
           cv2.imwrite('%s/%s.png'%(path,count), face_resize)
        count += 1
    else:
        message = "No Person Detected"
    cv2.putText(img, message, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Face Detection', img)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

print("Dataset obtained successfully.")

webcam.release()
cv2.destroyAllWindows()