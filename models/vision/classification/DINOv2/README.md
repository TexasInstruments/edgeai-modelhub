---
license: apache-2.0
tags:
- vision
- image-classification
- transformer
- self-supervised
datasets:
- imagenet-1k
---

<div align="center">

# DINOv2 for TI EdgeAI

### Self-Supervised Vision Transformer for Image Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**DINOv2** (Self-**Di**stillation with **No** Labels v2) produces high-performance visual features using a purely self-supervised training regime on 142M images — no labels required. The pretrained backbones are paired with a lightweight linear classification head for ImageNet-1K inference.

All models use a **ViT/14** patch size (14×14 patches) and are evaluated at **224×224** input resolution.

---

## Model Variants

| Model ID | Variant | Params | GFLOPs | Top-1 Acc | Edge Use |
|----------|---------|--------|--------|-----------|----------|
| `cl-mh6010` | ViT-S/14 distilled | 21M | 4.6 | **81.1%** | Recommended |
| `cl-mh6011` | ViT-B/14 distilled | 86M | 17.6 | **84.5%** | Feasible |
| `cl-mh6012` | ViT-L/14 distilled | 307M | 61.6 | **86.3%** | Not supported (TIDL) |
| `cl-mh6013` | ViT-S/14 distilled + registers | 21M | 4.6 | **80.9%** | Recommended |
| `cl-mh6014` | ViT-B/14 distilled + registers | 86M | 17.6 | **84.6%** | Feasible |
| `cl-mh6015` | ViT-L/14 distilled + registers | 307M | 61.6 | **86.7%** | Not supported (TIDL) |

> **With registers**: Vision Transformers with register tokens ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588)) exhibit fewer artifacts in feature maps and slightly better accuracy on dense prediction tasks. For pure classification, both variants perform comparably.

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
# Export recommended model (ViT-S/14, ~82MB)
python prepare_model.py --model dinov2_vits14_lc

# Export ViT-B/14 (~348MB)
python prepare_model.py --model dinov2_vitb14_lc

# Export ViT-S/14 with registers
python prepare_model.py --model dinov2_vits14_reg_lc

# Export ViT-B/14 with registers
python prepare_model.py --model dinov2_vitb14_reg_lc

# Export all edge-supported models
python prepare_model.py --model all
```

> **Note:** ViT-L/14 variants (`dinov2_vitl14_lc`, `dinov2_vitl14_reg_lc`) are excluded from edge deployment due to TIDL compilation failures and are not exported by `--model all`.

The script automatically:
- Downloads pretrained weights from PyTorch Hub (facebookresearch/dinov2)
- Exports backbone + linear classification head to ONNX (opset 17)
- Fixes dynamic shapes to static `[1, 3, 224, 224]`
- Validates and optionally simplifies the model graph

### 3. Compile for TI Hardware

**Using TIDL Runner (Recommended):**

```bash
tidlrunner-cli compile --target_device J784S4 \
  --config_path dinov2_vits14_lc_config.yaml
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
  --config_path dinov2_vits14_lc_config.yaml
```

---

## Preprocessing

All DINOv2 variants use standard ImageNet normalization:

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
| **Industrial** | Product classification, quality inspection |
| **Surveillance** | Scene understanding, object categorization |
| **Robotics** | Visual perception, environment understanding |
| **Agriculture** | Crop monitoring, disease detection |
| **Medical Imaging** | Feature extraction for downstream classifiers |

---

## Citations

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and others},
  journal={arXiv:2304.07193},
  year={2023}
}

@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```

---

## Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| **Registers Paper** | [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) |
| **Source Repo** | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

| Model | Task | Notes |
|-------|------|-------|
| ResNet-50 | Classification | Lightweight CNN baseline |
| CLIP | Classification + Text | Vision-language model |

---

<div align="center">

**License:** Apache 2.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

[Back to Model Hub](../../README.md)

</div>
