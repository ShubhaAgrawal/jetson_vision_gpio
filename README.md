# Jetson Vision GPIO — Edge AI Person Detection with TensorRT

An end-to-end computer vision pipeline running on **NVIDIA Jetson Orin Nano Super**, combining optimized AI inference, real-time hardware control, and networked data output.

Detects people via USB camera, controls a GPIO LED, and publishes live detection data to a Node-RED dashboard over MQTT.

---

## Results

| Method | Precision | Inference Time | Throughput | 
|--------|-----------|---------------|------------|
| YOLOv8n (PyTorch) | FP32 | 25.70 ms | 38.9 FPS |
| YOLOv8n (TensorRT) | FP16 | 11.42 ms | 87.5 FPS |

**2.25x speedup achieved through TensorRT FP16 optimization — same model, same hardware.**

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
├── vision_led.py          # Original project: Caffe SSD face detection + GPIO
├── baseline.py            # YOLOv8n PyTorch inference + GPIO (FP32 baseline)
├── trt_inference.py       # YOLOv8n TensorRT FP16 inference + GPIO
├── detection_mqtt.py      # Full pipeline: TensorRT + GPIO + MQTT publishing
├── benchmark.py           # Pure inference benchmark (PyTorch vs TensorRT)
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

Then open `http://<jetson-ip>:1880` in your browser and install the `node-red-dashboard` palette.

### 5. Export YOLOv8n to TensorRT

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True, device=0, simplify=True)
"
```

This takes approximately 7-10 minutes. The resulting `yolov8n.engine` file is optimized specifically for your Jetson hardware.

---

## Usage

### Run the benchmark (PyTorch vs TensorRT)
```bash
python3 benchmark.py
```

### Run full pipeline (TensorRT + GPIO + MQTT dashboard)
```bash
export DISPLAY=:0
python3 detection_mqtt.py
```

Then open `http://<jetson-ip>:1880/ui` to view the live dashboard.

---

## How TensorRT Optimization Works

The export pipeline converts the model through three stages:

```
YOLOv8n.pt (PyTorch) → yolov8n.onnx (ONNX) → yolov8n.engine (TensorRT FP16)
```

TensorRT spends the export time profiling hundreds of kernel configurations to find the fastest execution plan specifically for the Orin's GPU. FP16 (half precision) reduces weight memory by 50% and leverages the Orin's dedicated Tensor Cores for faster computation. The resulting `.engine` file is hardware-specific and permanently optimized — precision decisions are baked in at export time rather than evaluated at runtime.

---

## GPIO Wiring

| Jetson Pin | Connection |
|------------|------------|
| Pin 7 (BOARD) | LED anode via 220Ω resistor |
| Pin 6 (GND) | LED cathode |

---

## What I Learned

- TensorRT model export pipeline: PyTorch → ONNX → TensorRT engine
- FP16 vs FP32 precision tradeoffs on edge hardware
- MQTT publish/subscribe architecture for IoT data pipelines
- Node-RED for real-time dashboard visualization
- Measuring and interpreting inference benchmarks on embedded hardware
