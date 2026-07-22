---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

# YOLOX for TI EdgeAI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
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

This YOLOX model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready image detection with minimal setup.

YOLOX is an anchor-free version of YOLO with decoupled head design, providing strong performance with simpler architecture. It offers excellent accuracy-speed trade-offs across different model sizes.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

---
|Dataset |Model Name      |Model ID    |Input Size |mAP[.5:.95]% |Available |Notes                  |
|-       |-               |-           |-          |-            |-         |-                      |
|COCO    |yolox-nano      |`od-mh8003` |416×416    |24.8         | ✅       |Recommended, Smallest  |
|COCO    |yolox-tiny      |`od-mh8004` |416×416    |32.8         | ✅       |                       |
|COCO    |yolox-m         |`od-mh8005` |640×640    |46.9         | ✅       |                       |
|COCO    |yolox-l         |`od-mh8006` |640×640    |49.7         | ✅       |                       |
|COCO    |yolox-x         |`od-mh8007` |640×640    |51.2         | ✅       |Largest                |
|COCO    |yolox-darknet53 |`od-mh8008` |640×640    |47.4         | ✅       |Darknet53 backbone     |
---

## Setup & Installation

### Requirements
```bash
# Python dependencies
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
pip install onnxsim  # For model simplification
```
#### For TI hardware deployment
Refer the link for **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)**

## Usage

### Model Download
The models can be downloaded and prepared using the provided script.

Each variant has corresponding `.link` files and can be downloaded with `download_yolox.py`:

```bash
# Download specific variants
python download_yolox.py --model yolox_nano
python download_yolox.py --model yolox_tiny
python download_yolox.py --model yolox_m
python download_yolox.py --model yolox_l
python download_yolox.py --model yolox_x
python download_yolox.py --model yolox_darknet53

# Download multiple variants at once
python download_yolox.py --model yolox_nano yolox_tiny yolox_m

# Download with ONNX simplification (recommended)
python download_yolox.py --model yolox_nano --simplify

# Download and verify accuracy on COCO val2017
python download_yolox.py --model yolox_nano --verify --num-val-images 500
```

The script will automatically:
1. Download pre-built ONNX from GitHub releases (or convert from PyTorch if needed)
2. Fix batch dimensions to static shapes for hardware deployment
3. Optionally simplify the model using onnx-simplifier
4. Optionally verify accuracy on COCO val2017 dataset

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
# Should execute from tidlrunner setup directory.

# yolox-nano (416×416)
tidlrunner-cli compile --target_device J784S4 --config_path \
<edgeai-modelhub-path>/vision/detection/YOLOX/yolox_nano_416x416_config.yaml

# yolox-tiny (416×416)
tidlrunner-cli compile --target_device J784S4 --config_path \
<edgeai-modelhub-path>/vision/detection/YOLOX/yolox_tiny_416x416_config.yaml

# yolox-m (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path \
<edgeai-modelhub-path>/vision/detection/YOLOX/yolox_m_640x640_config.yaml

# yolox-l (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path \
<edgeai-modelhub-path>/vision/detection/YOLOX/yolox_l_640x640_config.yaml

# yolox-x (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path \
<edgeai-modelhub-path>/vision/detection/YOLOX/yolox_x_640x640_config.yaml

# yolox-darknet53 (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path \
<edgeai-modelhub-path>/vision/detection/YOLOX/yolox_darknet53_640x640_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

## Citation

If you use YOLOX in your research, please cite:

```bibtex
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

## Additional Resources

### Tools & Frameworks
- [EdgeAI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - Low-level compilation and inference
- [EdgeAI TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner) - High-level wrapper for easy deployment
- [EdgeAI Main Repository](https://github.com/TexasInstruments/edgeai) - Complete EdgeAI ecosystem

### YOLOX Resources
- [YOLOX Official Repository](https://github.com/Megvii-BaseDetection/YOLOX) - Original implementation by Megvii
- [YOLOX Paper](https://arxiv.org/abs/2107.08430) - arXiv paper with architecture details

**License:** apache-2.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** July 2026
