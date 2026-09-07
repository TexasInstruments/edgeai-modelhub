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

Pretrained weights are sourced from **torchvision** (BSD-3-Clause), trained on ImageNet-1K using a modernized training recipe.

All variants take a **224×224** input with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). The ideal pre-crop resize is variant-specific (236px for Tiny, 230px for Small, 232px for Base/Large) and is already set correctly in each variant's config YAML.

---

## Model Variants

| Model | Architecture | Params | GFLOPs | Top-1 Acc | Top-5 Acc | Validated Devices | Config |
|-------|-------------|--------|--------|-----------|-----------|--------------------|--------|
| `convnext_tiny`  | ConvNeXt-Tiny  | 28.6M  |  4.46 | **82.52%** | 96.15% | TDA4VH | [convnext_tiny_config.yaml](convnext_tiny_config.yaml) |
| `convnext_small` | ConvNeXt-Small | 50.2M  |  8.68 | **83.62%** | 96.65% | TDA4VH | [convnext_small_config.yaml](convnext_small_config.yaml) |
| `convnext_base`  | ConvNeXt-Base  | 88.6M  | 15.36 | **84.06%** | 96.87% | TDA4VH | [convnext_base_config.yaml](convnext_base_config.yaml) |
| `convnext_large` | ConvNeXt-Large | 197.8M | 34.36 | **84.41%** | 96.98% | TDA4VH | [convnext_large_config.yaml](convnext_large_config.yaml) |

**Recommended for edge deployment:** `convnext_tiny` delivers competitive accuracy (82.5%) at the lowest compute (4.46 GFLOPs, 28.6M params), making it the most practical choice for edge deployment. Larger variants offer incremental accuracy gains at significantly higher compute cost.

---

## Quick Start

### Prerequisites

```bash
pip install torch torchvision onnx>=1.22.0 onnxruntime>=1.23.2
# Optional but recommended for model optimization:
pip install onnx-simplifier
```

### Export the Model

```bash
# Export the default model (convnext_tiny)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model convnext_base

# Export all supported models
python prepare_model.py --model all

# List all available variants
python prepare_model.py --list-models

# Use a custom checkpoint
python prepare_model.py --model convnext_tiny --weights /path/to/checkpoint.pth
```

The script automatically:
- Downloads pretrained ImageNet-1K weights from torchvision (first run only)
- Exports the model to ONNX (opset 17) with static input shape [1, 3, 224, 224]
- Runs ONNX shape inference
- Optionally simplifies the graph with onnx-simplifier

### Compile and Infer uing edgeai-tidlrunner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using edgeai-tidlrunner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/convnext_tiny_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/convnext_tiny_config.yaml
```

### Compile and Infer using edgeai-tidl-tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

### Deploy using edgeai-tidl-tools:

Deplyment can be done using **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)**. For ONNX models, onnxruntime-tidl with TIDL acceleration can be used. Consult the documentation of edgeai-tidl-tools for more details.

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

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) |
| **Source Repo** | [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) |
| **Torchvision Docs** | [ConvNeXt](https://docs.pytorch.org/vision/main/models/convnext.html) |
| **HuggingFace** | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) |
| **edgeai-tidl-tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **edgeai-tidlrunner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

<table>
<tr>
<td align="center">

**ViT**
Pure Transformer
Attention-based

</td>
<td align="center">

**DINOv2**
Self-supervised ViT
Higher accuracy

</td>
<td align="center">

**DINO**
Self-supervised ViT
Linear head

</td>
<td align="center">

**ResNet**
CNN baseline
Lower compute

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
