---
license: bsd-3-clause
tags:
- vision
- image-classification
- transformer
datasets:
- imagenet-1k
---

<div align="center">

# ViT (Vision Transformer) for TI EdgeAI

### Pure Transformer for Image Classification at Scale

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**ViT** (Vision Transformer) applies the standard Transformer architecture directly to sequences of non-overlapping image patches — no convolutions. Introduced in [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., ICLR 2021), ViT demonstrates that a pure transformer pre-trained on large data transfers strongly to standard image recognition benchmarks.

Pretrained weights are sourced from **torchvision** (BSD-3-Clause), trained on ImageNet-1K using a DeiT-style recipe.

---

## Model Variants

| Model ID | Variant | Backbone | Patch Size | Params | GFLOPs | Top-1 Acc | Edge Use |
|---|---|---|---|---|---|---|---|
| `cl-mh6030` | `vit_b_16` | ViT-Base | 16×16 | 86.6M | 17.56 | **81.1%** | Recommended |
| `cl-mh6031` | `vit_b_32` | ViT-Base | 32×32 | 88.2M | 4.41 | **75.9%** | Feasible |
| `cl-mh6032` | `vit_l_16` | ViT-Large | 16×16 | 304.3M | 61.55 | **79.7%** | Feasible |
| `cl-mh6033` | `vit_l_32` | ViT-Large | 32×32 | 306.5M | 15.38 | **77.0%** | Feasible |

> **Recommended:** `vit_b_16` provides the best accuracy among all four variants (81.1%) at moderate compute (17.56 GFLOPs). `vit_b_32` is the lowest-compute option (4.41 GFLOPs) and a good choice for latency-constrained deployments.

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
# Export recommended model (vit_b_16, ~330 MB)
python prepare_model.py --model vit_b_16

# Export lightest model (vit_b_32, ~337 MB)
python prepare_model.py --model vit_b_32

# Export all variants
python prepare_model.py --model all

# List all available variants
python prepare_model.py --list-models

# Use a custom checkpoint
python prepare_model.py --model vit_b_16 --weights /path/to/checkpoint.pth
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
  --config_path vit_b_16_config.yaml
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
  --config_path vit_b_16_config.yaml
```

---

## Hardware Deployment

- **Primary Target:** TI J784S4 MPU
- **Framework:** TIDL Compilation Tools
- **Deployment:** [TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)

---

## Citation

```bibtex
@inproceedings{dosovitskiy2021image,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and
               Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and
               Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and
               Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2010.11929}
}
```

---

## Resources

| Resource | Link |
|---|---|
| **Paper** | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| **Source Repo** | [pytorch/vision](https://github.com/pytorch/vision) |
| **Torchvision Docs** | [VisionTransformer](https://docs.pytorch.org/vision/main/models/vision_transformer.html) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

| Model | Task | Notes |
|---|---|---|
| DINOv2 | Classification | Self-supervised ViT, higher accuracy |
| DINO | Classification | Self-supervised ViT with linear head |
| ResNet | Classification | CNN baseline, lower compute |
| MobileNetV3 | Classification | Lightweight CNN for edge |

---

<div align="center">

**License:** BSD-3-Clause  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

[Back to Model Hub](../../README.md)

</div>
