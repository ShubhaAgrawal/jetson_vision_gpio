import cv2
import time
import Jetson.GPIO as GPIO
from pathlib import Path
from ultralytics import YOLO

# ----------------------------
# Model paths (portable)
# ----------------------------

model = YOLO("yolov8n.pt")

# ----------------------------
# GPIO setup
# ----------------------------
LED_PIN = 7  # physical pin 7 (BOARD)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONF_THRESH = 0.5
PERSON_CLASS = 0
prev_time = time.time()
fps = 0.0

print("Running baseline YOLOv8n inference. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        h, w = frame.shape[:2]

        # --- Inference ---
        results = model(frame, conf = CONF_THRESH, classes = PERSON_CLASS, verbose=False)


        detections = results[0].boxes

        # --- LED control (every frame) ---
        person_detected = (len(detections) > 0)
        GPIO.output(LED_PIN, GPIO.HIGH if person_detected else GPIO.LOW)

        # --- Draw ---
        annoted = results[0].plot()

        # --- FPS ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt
        cv2.putText(annoted, f"FPS: {fps:.1f} Persons: {len(detections)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imshow("YOLOv8n Baseline", annoted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
