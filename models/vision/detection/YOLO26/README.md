---
license: agpl-3.0
tags:
- vision
- image-detection
datasets:
- COCO
---

# YOLO26 for TI EdgeAI

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

This YOLO26 model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready image detection with minimal setup.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details


| Dataset | Model Name | Model ID   | Input Size | mAP[.5:.95]% | Available | Notes |
|---------|------------|------------|------------|--------------|-----------|-------|
| COCO    | yolo26n    | `od-mh8015` | 640×640    | 57.5         | ✅        |       |
| COCO    | yolo26s    | `od-mh8016` | 640×640    | 60.0         | ✅        |       |
| COCO    | yolo26m    | `od-mh8017` | 640×640    | 62.0         | ✅        |       |
| COCO    | yolo26l    | `od-mh8018` | 640×640    | 64.0         | ✅        |       |
| COCO    | yolo26x    | `od-mh8019` | 640×640    | 66.0         | ✅        |       |
---

## Setup & Installation

### Requirements
```bash
# Python dependencies
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
```
#### For TI hardware deployment
Refer the link for **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)**

## Usage

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Download
The model can be downloaded automatically using the provided scripts.
Each variant has a corresponding `.link` file and can be prepared with `prepare_model.py`:

```bash
# Prepare a specific variant using --model flag
python prepare_model.py --model yolo26n
python prepare_model.py --model yolo26s
python prepare_model.py --model yolo26m
python prepare_model.py --model yolo26l
python prepare_model.py --model yolo26x

# Or specify the .link file directly
python prepare_model.py --link-file yolo26n.onnx.link
python prepare_model.py --link-file yolo26s.onnx.link
python prepare_model.py --link-file yolo26m.onnx.link
python prepare_model.py --link-file yolo26l.onnx.link
python prepare_model.py --link-file yolo26x.onnx.link
```

The `.link` file contains the HuggingFace model URL and will automatically:
1. Download the model if not present locally
2. Fix dynamic shapes to static shapes for hardware deployment
3. Validate the model structure

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

# Compile model
tidlrunner-cli compile --target_device J784S4 --config_path yolo26n_model_config.yaml

# Evaluate accuracy
tidlrunner-cli evaluate --target_device J784S4 --config_path yolo26n_model_config.yaml
```

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

## Additional Resources

### Tools & Frameworks
- [EdgeAI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - Low-level compilation and inference
- [EdgeAI TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner) - High-level wrapper for easy deployment
- [EdgeAI Main Repository](https://github.com/TexasInstruments/edgeai) - Complete EdgeAI ecosystem

**License:** agpl-3.0
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** June 2026
