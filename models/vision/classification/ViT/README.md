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

# ViT for TI EdgeAI

### Pure Transformer for Image Classification at Scale

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**ViT** (Vision Transformer) applies the standard Transformer architecture directly to sequences of non-overlapping image patches — no convolutions. Introduced in [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., ICLR 2021), ViT demonstrates that a pure transformer pre-trained on large data transfers strongly to standard image recognition benchmarks.

Pretrained weights are sourced from **torchvision** (BSD-3-Clause), trained on ImageNet-1K using a DeiT-style recipe. Each exported ONNX model is a single-input classification graph that outputs 1000-class ImageNet logits `[1, 1000]`.

---

## Model Variants

| Model | Architecture | Params | Top-1 Acc | Validated Devices | Config |
|-------|-------------|--------|-----------|--------------------|--------|
| `vit_b_16` | ViT-Base/16 | 86.6M | 81.1% | TDA4VH | [vit_b_16_config.yaml](vit_b_16_config.yaml) |
| `vit_b_32` | ViT-Base/32 | 88.2M | 75.9% | TDA4VH | [vit_b_32_config.yaml](vit_b_32_config.yaml) |
| `vit_l_16` | ViT-Large/16 | 304.3M | 79.7% | TDA4VH | [vit_l_16_config.yaml](vit_l_16_config.yaml) |
| `vit_l_32` | ViT-Large/32 | 306.5M | 77.0% | TDA4VH | [vit_l_32_config.yaml](vit_l_32_config.yaml) |

**Recommended for edge deployment:** `vit_b_16` (best accuracy/compute trade-off)

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
# Export the default model (vit_b_16)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model vit_b_32

# Export all supported models
python prepare_model.py --model all

# List all available variants
python prepare_model.py --list-models
```

The script automatically:
- Downloads pretrained ImageNet-1K weights from torchvision (first run only)
- Exports the model to ONNX (opset 17) with static input shape `[1, 3, 224, 224]`
- Runs ONNX shape inference
- Optionally simplifies the graph with onnx-simplifier

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/vit_b_16_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/vit_b_16_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

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

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| **Source Code** | [pytorch/vision](https://github.com/pytorch/vision) |
| **Torchvision Docs** | [VisionTransformer](https://docs.pytorch.org/vision/main/models/vision_transformer.html) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

<table>
<tr>
<td align="center">

**DINOv2**
Self-supervised ViT
Higher accuracy

</td>
<td align="center">

**DINO**
Self-supervised ViT
Linear classification head

</td>
<td align="center">

**ResNet**
CNN baseline
Lower compute

</td>
<td align="center">

**MobileNetV3**
Lightweight CNN
Built for edge

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
