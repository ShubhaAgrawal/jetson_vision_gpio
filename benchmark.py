import cv2
import time
from ultralytics import YOLO

model_pt = YOLO("yolov8n.pt")
model_fp16 = YOLO("yolov8n_fp16.engine", task="detect")
model_int8 = YOLO("yolov8n.engine", task = "detect")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

for _ in range(5):
    model_pt(frame, verbose=False)
    model_fp16(frame, verbose=False)
    model_int8(frame, verbose=False)

start = time.time()
for _ in range(50):
    model_pt(frame, conf=0.5, classes=[0], verbose=False)
pt_time = (time.time()-start) / 50 * 1000

start = time.time()
for _ in range(50):
    model_fp16(frame, conf=0.5, classes=[0], verbose=False)
fp16_time = (time.time()-start) / 50 * 1000

start = time.time()
for _ in range(50):
    model_int8(frame, conf=0.5, classes=[0], verbose=False)
int8_time = (time.time()-start)/50*1000

print("\n=========Benchmark Results========")
print(f"Pytorch_FP32 inference : {pt_time:.2f}ms ({1000/pt_time:.1f}FPS)")
print(f"TensorRT_FP16 inference : {fp16_time:.2f}ms ({1000/fp16_time:.1f}FPS)")
print(f"TensorRT_INT8 inference : {int8_time:.2f}ms ({1000/int8_time:.1f})FPS")
print(f"FP16 speedup over FP32 : {pt_time/fp16_time:.2f}x")
print(f"INT8 speed over FP32 : {pt_time/int8_time:.2f}x")
print(f"INT8 speedup over FP16 : {fp16_time/int8_time:.2f}x")
print("=====================================")
