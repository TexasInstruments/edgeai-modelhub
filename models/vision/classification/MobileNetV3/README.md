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

| Model | Architecture | Params | GFLOPs | Top-1 Acc | Top-5 Acc | Validated Devices | Config |
|-------|---------------|--------|--------|-----------|-----------|--------------------|--------|
| `mobilenetv3_large` | MobileNetV3-Large | 5.48M | 0.22 | **75.274%** | 92.566% | TDA4VH | [mobilenetv3_large_config.yaml](mobilenetv3_large_config.yaml) |
| `mobilenetv3_small` | MobileNetV3-Small | 2.54M | 0.06 | 67.668% | 87.402% | N/A | N/A |

`mobilenetv3_large` uses `IMAGENET1K_V2` weights (improved training recipe). `mobilenetv3_small` is excluded from `prepare_model.py`'s export catalog because it produces poor accuracy under TIDL compilation — the `mobilenetv3_small.onnx` bundled in this folder is provided for reference only and has no validated TIDL config.

**Recommended for edge deployment:** `mobilenetv3_large` (best accuracy/compute trade-off with a validated TIDL config)

---

## Quick Start

### Prerequisites

```bash
pip install torch torchvision onnx>=1.14.0 onnxruntime>=1.16.0
# Optional but recommended for model optimization:
pip install onnx-simplifier
```

### Export the Model

```bash
# Export the default model (MobileNetV3-Large)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model mobilenetv3_large

# Export with a custom input resolution
python prepare_model.py --model mobilenetv3_large --shape 224 224

# List all available variants
python prepare_model.py --list-models
```

The script automatically:
- Downloads pretrained ImageNet-1K weights from torchvision (`MobileNet_V3_Large_Weights.IMAGENET1K_V2`)
- Exports to ONNX (opset 17) with a static `[1, 3, 224, 224]` input shape
- Runs ONNX shape inference across all intermediate tensors
- Optionally simplifies the graph with onnxsim (use `--no-simplify` to skip)

> Note: `mobilenetv3_small` is currently excluded from the export catalog (poor accuracy under TIDL compilation), so `--model mobilenetv3_small` and `--model all` only produce `mobilenetv3_large`.

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/mobilenetv3_large_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/mobilenetv3_large_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use these models, please cite:

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

## 🔗 Resources

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

<table>
<tr>
<td align="center">

**ResNet**
Deeper CNN
Higher accuracy

</td>
<td align="center">

**ConvNeXt**
Modern CNN
ViT-inspired design

</td>
<td align="center">

**ViT**
Vision Transformer
Attention-based

</td>
<td align="center">

**DINOv2**
Self-supervised
Rich feature embeddings

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
