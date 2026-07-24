---
license: agpl-3.0
tags:
- vision
- image-detection
datasets:
- COCO
---

# YOLOv8 for TI EdgeAI

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange.svg)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green.svg)](https://github.com/TexasInstruments/edgeai)

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Citation](#citation)
- [Additional Resources](#additional-resources)

## Introduction

🚀 **Ready-to-deploy image detection for TI edge devices**

This YOLOv8 model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready image detection with minimal setup.

YOLOv8 is Ultralytics' state-of-the-art real-time object detector featuring an anchor-free split head design for improved accuracy and speed. It supports five size variants (n, s, m, l, x) offering a wide accuracy-speed trade-off, making it suitable for both resource-constrained edge devices and high-throughput deployments.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

---
| Dataset | Model Name | Model ID    | Input Size | mAP[.5:.95]% | Available | Notes                 |
| -       | -          | -           | -          | -            | -         | -                     |
| COCO    | yolov8n    | `od-mh8000` | 640×640    | 37.3         | ✅        | Recommended, Smallest |
| COCO    | yolov8s    | -           | 640×640    | 44.9         | -         |                       |
| COCO    | yolov8m    | `od-mh8001` | 640×640    | 50.2         | ✅        |                       |
| COCO    | yolov8l    | -           | 640×640    | 52.9         | -         |                       |
| COCO    | yolov8x    | -           | 640×640    | 53.9         | -         | Largest               |
---

## Setup & Installation

### Requirements
```bash
# Python dependencies
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
pip install ultralytics
```
#### For TI hardware deployment
Refer the link for **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)**

## Usage

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Export
Models are exported from Ultralytics PyTorch checkpoints to ONNX using the provided script.

```bash
# Export nano variant (default)
python ultralytics_yolo8_export.py

# Export specific variants
python ultralytics_yolo8_export.py --models yolov8n yolov8m

# Export all variants
python ultralytics_yolo8_export.py --models yolov8n yolov8s yolov8m yolov8l yolov8x

# Export with custom output directory
python ultralytics_yolo8_export.py --models yolov8n --output-dir ./exports
```

The script will automatically:
1. Install required dependencies (`onnx`, `ultralytics`)
2. Download the PyTorch checkpoint from Ultralytics
3. Export to ONNX format with static input shapes (opset 19)

### Using the model on TI device

#### Option 1: Advanced Users (TIDL Tools)
For users familiar with TIDL and requiring fine-grained control:

```bash
# Use edgeai-tidl-tools for compilation and inference
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tools
```

Refer to the [tidl-tools setup](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/README.md#getting-started) page for more details on compile and infer.

**Learn more:** [edgeai-tidl-tools documentation](https://github.com/TexasInstruments/edgeai-tidl-tools)

#### Option 2: Simplified Workflow (Recommended)
For easy compilation, benchmarking, and accuracy evaluation:

setup tidl runner using this link [edgeai-tidlrunner setup](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tidlrunner/docs/setup.md)

```bash
# Compile model and evaluate performance on J784S4.

# yolov8n
tidlrunner-cli compile --target_device J784S4 --config_path yolov8n_config.yaml

# yolov8m
tidlrunner-cli compile --target_device J784S4 --config_path yolov8m_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

## Additional Resources

### Tools & Frameworks
- [EdgeAI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - Low-level compilation and inference
- [EdgeAI TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner) - High-level wrapper for easy deployment
- [EdgeAI Main Repository](https://github.com/TexasInstruments/edgeai) - Complete EdgeAI ecosystem

### YOLOv8 Resources
- [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics) - Official implementation
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/) - Architecture details and usage guide

**License:** agpl-3.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** July 2026
