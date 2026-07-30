---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

# RTMDet for TI EdgeAI

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

This RTMDet model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready image detection with minimal setup.

RTMDet is a high-performance real-time object detector from OpenMMLab with a CSPNeXt backbone and an efficient anchor-free detection head. It achieves excellent accuracy-speed trade-offs across five model sizes (tiny, s, m, l, x), making it suitable for a wide range of deployment scenarios from resource-constrained edge devices to high-throughput server deployments.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

---
| Dataset | Model Name   | Model ID     | Input Size | mAP[.5:.95]% | Available | Notes              |
| -       | -            | -            | -          | -            | -         | -                  |
| COCO    | rtmdet-tiny  | `od-mh8020`  | 640×640    | 40.9         | ✅        | Recommended, Smallest |
| COCO    | rtmdet-s     | `od-mh8021`  | 640×640    | 44.5         | ✅        |                    |
| COCO    | rtmdet-m     | `od-mh8022`  | 640×640    | 49.3         | ✅        |                    |
| COCO    | rtmdet-l     | `od-mh8023`  | 640×640    | 51.4         | ✅        |                    |
| COCO    | rtmdet-x     | `od-mh8024`  | 640×640    | 52.8         | ✅        | Largest            |
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

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Download
The models can be downloaded and converted to ONNX using the provided script.

Each variant has a corresponding `.link` file and can be downloaded with `prepare_model.py`:

```bash
# Download specific variants
python prepare_model.py --models tiny
python prepare_model.py --models s
python prepare_model.py --models m
python prepare_model.py --models l
python prepare_model.py --models x

# Download multiple variants at once
python prepare_model.py --models tiny s m

# Download all variants
python prepare_model.py

# Download with ONNX simplification (recommended)
python prepare_model.py --models tiny --simplify
```

The script will automatically:
1. Install all required dependencies
2. Download the PyTorch checkpoint from OpenMMLab
3. Download the model configuration files
4. Convert to ONNX format
5. Fix batch dimensions to static shapes for hardware deployment
6. Optionally simplify the model using onnx-simplifier

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

# rtmdet-tiny (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtmdet_tiny_config.yaml

# rtmdet-s (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtmdet_s_config.yaml

# rtmdet-m (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtmdet_m_config.yaml

# rtmdet-l (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtmdet_l_config.yaml

# rtmdet-x (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtmdet_x_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

## Citation

If you use RTMDet in your research, please cite:

```bibtex
@article{lyu2022rtmdet,
  title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
  author={Lyu, Chengqi and Zhang, Wenwei and Huang, Haian and Zhou, Yue and Wang, Yudong and Liu, Yanyi and Zhang, Shilong and Chen, Kai},
  journal={arXiv preprint arXiv:2212.07784},
  year={2022}
}
```

## Additional Resources

### Tools & Frameworks
- [EdgeAI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - Low-level compilation and inference
- [EdgeAI TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner) - High-level wrapper for easy deployment
- [EdgeAI Main Repository](https://github.com/TexasInstruments/edgeai) - Complete EdgeAI ecosystem

### RTMDet Resources
- [RTMDet Official Repository](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) - OpenMMLab implementation
- [RTMDet Paper](https://arxiv.org/abs/2212.07784) - arXiv paper with architecture details

**License:** apache-2.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** July 2026
