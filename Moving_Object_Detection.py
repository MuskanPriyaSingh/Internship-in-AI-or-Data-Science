import cv2 
import time
import imutils
import pygame
import threading

pygame.mixer.init()

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 5
object_count = 0  # Initialize object count
sound_playing = False
stop_sound = threading.Event()

def play_sound():
    global sound_playing
    try:
        sound = pygame.mixer.Sound('alert.wav')
        sound.play()
        while sound.get_length() > 0 and not stop_sound.is_set():
            pygame.time.Clock().tick(10)  # Adjust playback speed if needed
    except Exception as e:
        print(f"Error playing sound: {e}")
    finally:
        sound.stop()
        sound_playing = False

while True:
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gaussianImg
        continue
    
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    detected = False  # Flag to check if an object is detected
    
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detected = True
        text = "Moving Object detected"
    
    if detected:
        if not sound_playing:
            stop_sound.clear()
            sound_thread = threading.Thread(target=play_sound)
            sound_thread.start()
            sound_playing = True
        object_count += 1  # Increment the count when an object is detected
        print(f"Moving Object detected! Total Count: {object_count}")
    else:
        if sound_playing:
            stop_sound.set()
            sound_thread.join()
            sound_thread = None
            sound_playing = False
    
    text2 = f"Object Count: {object_count}"
    
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("cameraFeed", img)
    
    # Update firstFrame periodically to detect ongoing movements
    firstFrame = gaussianImg
    
    key = cv2.waitKey(1) & 0xF
    if key == ord("q"):
        break

if sound_thread is not None and sound_thread.is_alive():
    stop_sound.set()
    sound_thread.join()

cam.release()
cv2.destroyAllWindows

