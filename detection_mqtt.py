import cv2
import time
import Jetson.GPIO as GPIO
from ultralytics import YOLO
import json
import paho.mqtt.client as mqtt

# --------------------------
# MQTT Setup
# --------------------------
MQTT_Broker = "localhost"
MQTT_Topic = "jetson/detection"

client = mqtt.Client()
client.connect(MQTT_Broker,1883,60)
client.loop_start()

# ---------------------------
# GPIO Setup
# ---------------------------
LED_PIN = 7
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# ---------------------------
# Load TensorRT Engine
# ---------------------------
model = YOLO("yolov8n.engine", task="detect")

# ---------------------------
# Camera
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONF_THRESH = 0.5
PERSON_CLASS = 0

prev_time = time.time()
fps = 0.0

print("Running TesnorRT inference with MQTT, Press Q to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera Error")
            break

        # -------Inference-------
        results = model(frame, conf=CONF_THRESH, classes=[PERSON_CLASS], verbose=False)
        detections = results[0].boxes
        person_detected = (len(detections)>0)

        # -------GPIO------------
        GPIO.output(LED_PIN, GPIO.HIGH if person_detected else GPIO.LOW)

        # ----------FPS----------
        now = time.time()
        dt = now-prev_time
        prev_time = now
        if dt>0:
            fps = 1.0/dt
        
        # ------Get Best Confidence Score--------
        confidence = 0.0
        if person_detected:
            confidences = detections.conf.tolist()
            confidence = round(max(confidences)*100,1)

        # --------Publish to MQTT--------
        payload = json.dumps({
            "count": len(detections),
            "confidence":confidence,
            "fps":round(fps,1),
            "person_detected":person_detected
            })
        client.publish(MQTT_Topic,payload)

        # ----------Draw-----------
        annotated = results[0].plot()
        cv2.putText(annotated, f"TRT FP16 | FPS: {fps:.1f} Persons: {len(detections)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2 )
        cv2.imshow("Jetson Detection + MQTT", annotated)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    client.loop_stop()
    client.disconnect()