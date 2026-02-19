# Jetson Vision GPIO

This project uses NVIDIA Jetson Orin Nano to control an LED using computer vision.

The script captures video from a camera and turns a GPIO pin ON or OFF based on detection.

---

## Requirements

Hardware:
- NVIDIA Jetson Orin Nano
- USB Camera
- LED with resistor
- Breadboard and jumper wires

Software:
- Python 3
- OpenCV
- Jetson.GPIO

Install dependencies:

```bash
pip install opencv-python Jetson.GPIO
