import cv2
import os

#Create directory to store images
os.makedirs("calibration_images", exist_ok=True) 

#SEtting up the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Initiating counters
count = 0
Total = 200

while count < Total:
    ret, frame = cap.read()
    if not ret:
        print("Camera Error")
        break

    filename = f"calibration_images/frame_{count:04d}.jpg"
    cv2.imwrite(filename,frame)
    count+=1

    if count%20 == 0:
        print(f"Collected {count}/{Total} frames")

cap.release()
print(f"Done. {count} frame saved to calibration_images/")