---
license: bsd-3-clause
tags:
- vision
- image-classification
- cnn
- mobile
datasets:
- imagenet-1k
---

<div align="center">

# MobileNetV3 for TI EdgeAI

### Efficient Mobile CNN for Image Classification

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**MobileNetV3** ([Searching for MobileNetV3](https://arxiv.org/abs/1905.02244), Howard et al., 2019) combines hardware-aware Neural Architecture Search (NAS) with NetAdapt and a redesigned last stage to deliver state-of-the-art accuracy for mobile and edge inference. Key improvements over MobileNetV2 include hard-swish activations, squeeze-and-excitation modules in the bottleneck layers, and an optimized final classifier.

Both variants are evaluated at **224×224** input resolution on **ImageNet-1K** and distributed via [torchvision](https://pytorch.org/vision/stable/models/mobilenetv3.html).

---

## Model Variants

| Model ID | Variant | Params | GFLOPs | Top-1 Acc | Top-5 Acc |
|----------|---------|--------|--------|-----------|-----------|
| `cl-mh6025` | MobileNetV3-Large | 5.48M | 0.22 | **75.274%** | 92.566% |

Uses `IMAGENET1K_V2` weights (improved training recipe).

---

## Quick Start

### 1. Prerequisites

```bash
pip install torch torchvision onnx>=1.14.0 onnxruntime>=1.16.0
# Optional but recommended for model optimization:
pip install onnx-simplifier
```

### 2. Export Model to ONNX

```bash
# Export MobileNetV3-Large (~21 MB)
python prepare_model.py --model mobilenetv3_large

# List all available variants
python prepare_model.py --list-models
```

The script automatically:
- Downloads pretrained ImageNet-1K weights from torchvision
- Exports to ONNX (opset 17) with static `[1, 3, 224, 224]` input shape
- Runs ONNX shape inference across all intermediate tensors
- Optionally simplifies the graph with onnxsim (use `--no-simplify` to skip)

### 3. Compile for TI Hardware

**Using TIDL Runner (Recommended):**

```bash
tidlrunner-cli compile --target_device J784S4 \
  --config_path mobilenetv3_large_config.yaml
```

**Using TIDL Tools (Advanced):**

```bash
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tools
# Follow setup at https://github.com/TexasInstruments/edgeai-tidl-tools
```

### 4. Evaluate Performance

```bash
tidlrunner-cli evaluate --target_device J784S4 \
  --config_path mobilenetv3_large_config.yaml
```

---

## Preprocessing

MobileNetV3-Large uses standard ImageNet normalization:

| Parameter | Value |
|-----------|-------|
| Resize | 256 (short side) |
| Crop | 224×224 (center) |
| Layout | NCHW |
| Mean (BGR) | `[123.675, 116.28, 103.53]` |
| Scale | `[0.017125, 0.017507, 0.017429]` |
| Channels | RGB (reverse_channels: false) |

---

## Hardware Deployment

- **Primary Target:** TI J784S4 MPU
- **Framework:** TIDL Compilation Tools
- **Deployment:** [TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)

---

## Use Cases

| Application | Details |
|-------------|---------|
| **Mobile / Embedded** | On-device classification with tight latency and power budgets |
| **Industrial** | Product inspection, defect detection, sorting |
| **Surveillance** | Scene classification, object categorization |
| **Robotics** | Fast visual perception for navigation and manipulation |
| **IoT** | Always-on classification in resource-constrained sensors |

---

## Citation

```bibtex
@inproceedings{Howard2019MobileNetV3,
  title     = {Searching for MobileNetV3},
  author    = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh
               and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun
               and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

---

## Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:1905.02244](https://arxiv.org/abs/1905.02244) |
| **PyTorch Docs** | [torchvision MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html) |
| **Source Code** | [pytorch/vision](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

| Model | Task | Notes |
|-------|------|-------|
| ResNet-50 | Classification | Deeper CNN, higher accuracy |
| DINOv2 | Classification | ViT-based, self-supervised features |
| DINO | Classification | ViT-based self-supervised baseline |

---

<div align="center">

**License:** BSD-3-Clause  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

[Back to Model Hub](../../README.md)

</div>
