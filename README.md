<div align="center">

# 🚀 TI EdgeAI Model Hub

### Computer Vision Models for TI Edge Devices

[![License](https://img.shields.io/badge/License-Mixed%20(Apache%202.0%20%26%20AGPL%203.0)-brightgreen?style=for-the-badge)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![TI EdgeAI](https://img.shields.io/badge/TI-EdgeAI-red?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge)](https://www.python.org/)

</div>

---

## 📋 Overview

TI EdgeAI Model Hub provides a curated collection of **pre-trained, hardware-optimized models** for computer vision tasks on Texas Instruments MPU devices. All models are:

✅ **Hardware-Optimized** - Compiled for TI MPU processors  
✅ **Production-Ready** - Validated and benchmarked on real hardware  
✅ **Easy to Deploy** - Automated scripts for model preparation  
✅ **Open Source** - Apache 2.0 & AGPL 3.0 licenses  

---

## 🎯 Use Cases

<table>
<tr>
<td>🏭 <b>Industrial Automation</b><br/>Quality inspection & defect detection</td>
<td>📹 <b>Smart Surveillance</b><br/>Real-time video analysis</td>
</tr>
<tr>
<td>🤖 <b>Robotics</b><br/>Autonomous navigation & control</td>
<td>🌐 <b>IoT Vision</b><br/>Edge AI applications</td>
</tr>
</table>

---

## 🛠️ Compilation & Deployment

Models are compiled and deployed using:

| Tool | Purpose | Use Case |
|------|---------|----------|
| **[TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner)** | 🎯 High-level compilation interface | Recommended for most users |
| **[TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools)** | ⚙️ Low-level compilation framework | Advanced users & custom workflows |

---

## 📦 Model Catalog

| Model | Capabilities | Variants | Input | Performance | License | Docs |
|-------|--------------|----------|-------|-------------|---------|------|
| **ResNet-50** | 🖼️ Image Classification | resNet50 (cl-mh6000), resnet50-v1-7 (cl-mh6001) | 224×224 | 76.15% Top-1 | Apache 2.0 | [📖](models/vision/classification/resnet/) |
| **DINO** | 🖼️ Image Classification | ViT-S/16, ViT-S/8, ViT-B/16, ViT-B/8, ResNet-50 | 224×224 | 75.3–80.1% Top-1 | Apache 2.0 | [📖](models/vision/classification/DINO/) |
| **DINOv2** | 🖼️ Image Classification | ViT-S/14, ViT-B/14 (w/ & w/o registers) | 224×224 | 80.9–84.6% Top-1 | Apache 2.0 | [📖](models/vision/classification/DINOv2/) |
| **DEIMv2** | 🎯 Object Detection | atto, femto, pico, n, s, m, l, x | 320–640px | 23.8–57.8% mAP | Apache 2.0 | [📖](models/vision/detection/DEIMv2/) |
| **DETR** | 🎯 Object Detection | detr_resnet50, detr_resnet50_dc5, detr_resnet101, detr_resnet101_dc5 | 800×800 (flexible) | AP50:95 42.0-44.9, AP50 62.4-64.7 (detection) | Apache 2.0 | [📖](models/vision/detection/DETR/) |
| **Deformable-DETR** | 🎯 Object Detection | single-scale, multi-scale, +bbox refine, two-stage | 800×800 | AP50:95 39.4-46.9% | Apache 2.0 | [📖](models/vision/detection/Deformable-DETR/) |
| **RF-DETR** | 🎯 Object Detection | nano, s, m, l | 384–704px | 48.4-56.5% mAP | Apache 2.0 | [📖](models/vision/detection/RF-DETR/) |
| **RT-DETRv2** | 🎯 Object Detection | s, ms, m, l, x | 640×640 | 48.1-54.3% mAP | Apache 2.0 | [📖](models/vision/detection/RT-DETRv2/) |
| **RTMDet** | 🎯 Object Detection | tiny, s, m, l, x | 640×640 | 40.9-52.8% mAP | Apache 2.0 | [📖](models/vision/detection/RTMDet/) |
| **YOLO11** | 🎯 Object Detection | n, s, m, l, x | 640×640 | 39.5-54.7% mAP | AGPL 3.0 | [📖](models/vision/detection/YOLO11/) |
| **YOLO26** | 🎯 Object Detection | n, s, m, l, x | 640×640 | 57.5-66.0% mAP | AGPL 3.0 | [📖](models/vision/detection/YOLO26/) |
| **YOLOv8** | 🎯 Object Detection | n, m | 640×640 | COCO trained | AGPL 3.0 | [📖](models/vision/detection/YOLOv8/) |
| **YOLOX** | 🎯 Object Detection | nano, tiny, m, l, x, darknet53 | 416×416/640×640 | 24.8-51.2% mAP | Apache 2.0 | [📖](models/vision/detection/YOLOX/) |

---

## 🚀 Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/TexasInstruments/edgeai-modelhub.git
cd edgeai-modelhub
```

### 2️⃣ Install Dependencies
```bash
pip install onnx>=1.22.0 onnxruntime>=1.23.2
```

### 3️⃣ Download & Prepare Model
```bash
# Choose your model folder
cd models/vision/detection/YOLO11/

# Prepare model (auto-downloads from HuggingFace)
python prepare_model.py --model yolo11n
```

### 4️⃣ Compile for TI Hardware
```bash
# Use TIDL Runner (recommended)
tidlrunner-cli compile --target_device J784S4 \
  --config_path yolo11n_model_config.yaml
```

---

## 📚 Documentation

Detailed setup, usage, and deployment guides for each model:

| Model | Documentation |
|-------|---------------|
| 🖼️ ResNet-50 Classification | [Read](models/vision/classification/resnet/README.md) |
| 🖼️ DINO Classification | [Read](models/vision/classification/DINO/README.md) |
| 🖼️ DINOv2 Classification | [Read](models/vision/classification/DINOv2/README.md) |
| 🎯 DEIMv2 Detection | [Read](models/vision/detection/DEIMv2/README.md) |
| 🎯 DETR Detection | [Read](models/vision/detection/DETR/README.md) |
| 🎯 Deformable-DETR Detection | [Read](models/vision/detection/Deformable-DETR/README.md) |
| 🎯 RF-DETR Detection | [Read](models/vision/detection/RF-DETR/README.md) |
| 🎯 RT-DETRv2 Detection | [Read](models/vision/detection/RT-DETRv2/README.md) |
| 🎯 RTMDet Detection | [Read](models/vision/detection/RTMDet/README.md) |
| 🎯 YOLO11 Detection | [Read](models/vision/detection/YOLO11/README.md) |
| 🎯 YOLO26 Detection | [Read](models/vision/detection/YOLO26/README.md) |
| 🎯 YOLOv8 Detection | [Read](models/vision/detection/YOLOv8/README.md) |
| 🎯 YOLOX Detection | [Read](models/vision/detection/YOLOX/README.md) |

---

## 🔗 Resources & Links

<table>
<tr>
<td align="center" width="33%">

### 🌐 Ecosystem
[TI EdgeAI](https://github.com/TexasInstruments/edgeai)  
Main repository & documentation

</td>
<td align="center" width="33%">

### 🔧 Tools
[TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner)  
[TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

</td>
<td align="center" width="33%">

### 📖 Formats
[ONNX](https://onnx.ai/)  
[EdgeAI SDK](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)

</td>
</tr>
</table>

---

## 📋 Supported Hardware

- 🎯 **TI J784S4** MPU (Primary)
- 🎯 Other TI EdgeAI-supported devices with TIDL compilation

---

## 📜 License

- 📄 **ResNet-50, DINO, DINOv2, DEIMv2, DETR, Deformable-DETR, RF-DETR, RT-DETRv2, RTMDet, YOLOX**: Apache 2.0
- 📄 **YOLO11, YOLO26, YOLOv8**: AGPL 3.0

---

<div align="center">

### ⭐ If you find this useful, please consider starring the repository!

**Maintained by** Texas Instruments EdgeAI Team  
**Last Updated** July 2026

[📧 Contact](https://github.com/TexasInstruments/edgeai) • [🐛 Issues](https://github.com/TexasInstruments/edgeai/issues) • [💬 Discussions](https://github.com/TexasInstruments/edgeai/discussions)

</div>
