import cv2
import time
import Jetson.GPIO as GPIO
from pathlib import Path

# ----------------------------
# Model paths (portable)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PROTO = str(BASE_DIR / "models" / "deploy.prototxt")
MODEL_WEIGHTS = str(BASE_DIR / "models" / "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

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

CONF_THRESH = 0.5
prev_time = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        h, w = frame.shape[:2]

        # --- Inference ---
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), swapRB=False)
        net.setInput(blob)
        raw = net.forward()

        detections = []
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            if conf > CONF_THRESH:
                x1n = float(raw[0, 0, i, 3])
                y1n = float(raw[0, 0, i, 4])
                x2n = float(raw[0, 0, i, 5])
                y2n = float(raw[0, 0, i, 6])
                detections.append((x1n, y1n, x2n, y2n, conf))

        # --- LED control (every frame) ---
        person_detected = (len(detections) > 0)
        GPIO.output(LED_PIN, GPIO.HIGH if person_detected else GPIO.LOW)

        # --- Draw ---
        for (x1n, y1n, x2n, y2n, conf) in detections:
            x1 = int(x1n * w)
            y1 = int(y1n * h)
            x2 = int(x2n * w)
            y2 = int(y2n * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, f"FACE {conf:.2f}", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        # --- FPS ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imshow("Jetson Vision + LED", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
