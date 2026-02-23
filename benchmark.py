import cv2
import time
from ultralytics import YOLO

model_pt = YOLO("yolov8n.pt")
model_trt = YOLO("yolov8n.engine", task="detect")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

for _ in range(5):
    model_pt(frame, verbose=False)
    model_trt(frame, verbose=False)

start = time.time()
for _ in range(50):
    model_pt(frame, conf=0.5, classes=[0], verbose=False)
pt_time = (time.time()-start) / 50 * 1000

start = time.time()
for _ in range(50):
    model_trt(frame, conf=0.5, classes=[0], verbose=False)
trt_time = (time.time()-start) / 50 * 1000

print(f"Pytorch inference : {pt_time:.2f}ms ({1000/pt_time:.1f}FPS)")
print(f"TensorRT inference : {trt_time:.2f}ms ({1000/trt_time:.1f}FPS)")
print(f"Speedup : {pt_time/trt_time:.2f}x")
