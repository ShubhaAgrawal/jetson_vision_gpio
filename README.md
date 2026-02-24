# Jetson Vision GPIO — Edge AI Person Detection with TensorRT

An end-to-end computer vision pipeline running on **NVIDIA Jetson Orin Nano Super**, combining optimized AI inference, real-time hardware control, and networked data output.

Detects people via USB camera, controls a GPIO LED, and publishes live detection data to a Node-RED dashboard over MQTT.

---

## Benchmark Results

| Method | Precision | Inference Time | Throughput | Speedup |
|--------|-----------|---------------|------------|---------|
| PyTorch | FP32 | 26.28 ms | 38.1 FPS | 1.00x |
| TensorRT | FP16 | 13.56 ms | 73.8 FPS | 1.94x |
| TensorRT | INT8 | 14.47 ms | 69.1 FPS | 1.82x |

**TensorRT FP16 achieves the best result — 1.94x faster than PyTorch baseline.**

> **Note on INT8 vs FP16:** INT8 is marginally slower than FP16 on the Orin Nano for this model. This is because YOLOv8n is a small model and the Orin's Tensor Cores are heavily optimized for FP16. The per-layer quantization/dequantization overhead in INT8 outweighs the speed gain at this model size. Larger models (YOLOv8l, YOLOv8x) would likely show INT8 winning clearly. This highlights an important real-world consideration: **optimization results are hardware and model-size dependent.**

---

## System Architecture

```
USB Camera
    │
    ▼
TensorRT FP16 Inference (YOLOv8n)
    │
    ├──► GPIO Pin 7 ──► LED (ON when person detected)
    │
    └──► MQTT Broker (Mosquitto)
              │
              ▼
         Node-RED Dashboard
         (live person count, confidence gauge, FPS chart)
```

---

## Hardware

- NVIDIA Jetson Orin Nano Super Developer Kit
- USB Camera
- LED + 220Ω resistor
- Breadboard and jumper wires

---

## Software Stack

- **JetPack** 6.1 (L4T R36.4.7)
- **Python** 3.10
- **PyTorch** 2.5.0 (Jetson build)
- **Ultralytics** YOLOv8 8.4.14
- **TensorRT** 10.3.0
- **Mosquitto** MQTT broker
- **Node-RED** dashboard
- **OpenCV** 4.x
- **Jetson.GPIO**

---

## Project Structure

```
jetson_vision_gpio/
├── vision_led.py              # Original project: Caffe SSD face detection + GPIO
├── baseline.py                # YOLOv8n PyTorch inference + GPIO (FP32 baseline)
├── trt_inference.py           # YOLOv8n TensorRT FP16 inference + GPIO
├── detection_mqtt.py          # Full pipeline: TensorRT + GPIO + MQTT publishing
├── benchmark.py               # Three-way benchmark: PyTorch FP32 vs TensorRT FP16 vs INT8
├── collect_calibration.py     # Captures calibration frames for INT8 export
├── calibration.yaml           # Dataset config for INT8 calibration
├── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
└── README.md
```

---

## Setup

### 1. Install PyTorch for JetPack 6.1

```bash
# Install cuSPARSELt dependency
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
tar xf libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/include/* /usr/local/cuda/include/
sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig

# Install PyTorch
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Downgrade numpy for compatibility
pip3 install "numpy<2"
```

### 2. Install Python dependencies

```bash
pip3 install ultralytics paho-mqtt "onnx<2.0.0" onnxslim
```

### 3. Install and start Mosquitto

```bash
sudo apt-get install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

### 4. Install and start Node-RED

```bash
bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)
sudo systemctl enable nodered
sudo systemctl start nodered
```

Then open `http://<jetson-ip>:1880` in your browser and install the `node-red-dashboard` palette via Manage Palette.

### 5. Export YOLOv8n to TensorRT FP16

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True, device=0, simplify=True)
"
```

Rename the output so it isn't overwritten later:
```bash
mv yolov8n.engine yolov8n_fp16.engine
```

### 6. Collect INT8 calibration images

```bash
python3 collect_calibration.py
```

Move around in front of the camera during collection to capture varied frames.

### 7. Export YOLOv8n to TensorRT INT8

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', int8=True, data='calibration.yaml', device=0, simplify=True)
"
```

Both exports take 7-15 minutes each as TensorRT profiles your specific hardware.

---

## Usage

### Run the three-way benchmark
```bash
python3 benchmark.py
```

### Run full pipeline (TensorRT FP16 + GPIO + MQTT + dashboard)
```bash
export DISPLAY=:0
python3 detection_mqtt.py
```

Then open `http://<jetson-ip>:1880/ui` to view the live dashboard.

---

## How TensorRT Optimization Works

The export pipeline converts the model through three stages:

```
yolov8n.pt (PyTorch) → yolov8n.onnx (ONNX) → yolov8n.engine (TensorRT)
```

TensorRT spends the export time profiling hundreds of kernel configurations to find the fastest execution plan specifically for the Orin's GPU. The resulting `.engine` file is hardware-specific — precision decisions are baked in at export time rather than evaluated at runtime every frame.

**FP16** halves the memory footprint of weights and leverages the Orin's dedicated Tensor Cores for faster computation with minimal accuracy loss.

**INT8** goes further to 8-bit integers but requires a calibration dataset so TensorRT can learn the optimal quantization mapping per layer without sacrificing accuracy.

---

## GPIO Wiring

| Jetson Pin | Connection |
|------------|------------|
| Pin 7 (BOARD) | LED anode via 220Ω resistor |
| Pin 6 (GND) | LED cathode |

---

## What I Learned

- TensorRT model export pipeline: PyTorch → ONNX → TensorRT engine
- FP32 vs FP16 vs INT8 precision tradeoffs on edge hardware
- Why INT8 doesn't always outperform FP16 — model size and hardware architecture matter
- MQTT publish/subscribe architecture for IoT data pipelines
- Node-RED for real-time dashboard visualization
- Measuring and interpreting inference benchmarks on embedded hardware
- Importance of calibration data quality for INT8 quantization
