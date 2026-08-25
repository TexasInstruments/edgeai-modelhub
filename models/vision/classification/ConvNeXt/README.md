---
license: bsd-3-clause
tags:
- vision
- image-classification
- convnet
datasets:
- imagenet-1k
---

<div align="center">

# ConvNeXt for TI EdgeAI

### A ConvNet for the 2020s — Pure ConvNet Matching Transformer Accuracy

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**ConvNeXt** is a pure convolutional network modernized by incorporating design principles from Vision Transformers (Swin Transformer). Introduced in [*A ConvNet for the 2020s*](https://arxiv.org/abs/2201.03545) (Liu et al., CVPR 2022), ConvNeXt matches or surpasses Swin Transformers in accuracy while retaining the simplicity, efficiency, and hardware-friendliness of standard CNNs — no attention mechanisms, no positional encodings.

Pretrained weights are sourced from **torchvision** (BSD-3-Clause), trained on ImageNet-1K using a modernized recipe.

---

## Model Variants

| Variant | Backbone | Params | GFLOPs | Top-1 Acc | Top-5 Acc | Edge Use |
|---|---|---|---|---|---|---|
| `convnext_tiny`  | ConvNeXt-Tiny  | 28.6M  |  4.46 | **82.52%** | 96.15% | Recommended |
| `convnext_small` | ConvNeXt-Small | 50.2M  |  8.68 | **83.62%** | 96.65% | Feasible |
| `convnext_base`  | ConvNeXt-Base  | 88.6M  | 15.36 | **84.06%** | 96.87% | Feasible |
| `convnext_large` | ConvNeXt-Large | 197.8M | 34.36 | **84.41%** | 96.98% | Feasible |

> **Recommended:** `convnext_tiny` delivers competitive accuracy (82.5%) at the lowest compute (4.46 GFLOPs, 28.6M params), making it the most practical choice for edge deployment. Larger variants offer incremental accuracy gains at significantly higher compute cost.

All variants use a **224×224** input resolution with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

---

## Quick Start

### 1. Prerequisites

```bash
pip install torch torchvision onnx>=1.22.0 onnxruntime>=1.23.2
# Optional but recommended for model optimization:
pip install onnx-simplifier
```

### 2. Export Model to ONNX

```bash
# Export recommended model (convnext_tiny, ~109 MB)
python prepare_model.py --model convnext_tiny

# Export a heavier variant
python prepare_model.py --model convnext_base

# Export all variants
python prepare_model.py --model all

# List all available variants
python prepare_model.py --list-models

# Use a custom checkpoint
python prepare_model.py --model convnext_tiny --weights /path/to/checkpoint.pth
```

The script automatically:
- Downloads pretrained ImageNet-1K weights from torchvision (first run only)
- Exports the model to ONNX (opset 17) with static shape `[1, 3, 224, 224]`
- Runs ONNX shape inference
- Optionally simplifies the graph with onnxsim

### 3. Compile for TI Hardware

**Using TIDL Runner (Recommended):**

```bash
tidlrunner-cli compile --target_device J784S4 \
  --config_path convnext_tiny_config.yaml
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
  --config_path convnext_tiny_config.yaml
```

---

## Hardware Deployment

- **Primary Target:** TI J784S4 MPU
- **Framework:** TIDL Compilation Tools
- **Deployment:** [TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)

---

## Preprocessing Details

| Step | Value |
|---|---|
| Resize | Variant-specific (see below) |
| Center Crop | 224×224 |
| Normalization mean | [0.485, 0.456, 0.406] |
| Normalization std | [0.229, 0.224, 0.225] |
| Data layout | NCHW |

| Variant | Resize (before crop) |
|---|---|
| `convnext_tiny` | 236 |
| `convnext_small` | 230 |
| `convnext_base` | 232 |
| `convnext_large` | 232 |

---

## Citation

```bibtex
@inproceedings{liu2022convnet,
  title     = {A ConvNet for the 2020s},
  author    = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and
               Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.03545}
}
```

---

## Resources

| Resource | Link |
|---|---|
| **Paper** | [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) |
| **Source Repo** | [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) |
| **Torchvision Docs** | [ConvNeXt](https://docs.pytorch.org/vision/main/models/convnext.html) |
| **HuggingFace** | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

| Model | Task | Notes |
|---|---|---|
| ViT | Classification | Pure Transformer, attention-based |
| DINOv2 | Classification | Self-supervised ViT, higher accuracy |
| DINO | Classification | Self-supervised ViT with linear head |
| ResNet | Classification | CNN baseline, lower compute |

---

<div align="center">

**License:** BSD-3-Clause  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

[Back to Model Hub](../../README.md)

</div>
