---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
---

<div align="center">

# ResNet for TI EdgeAI

### Deep Residual Network for Image Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**ResNet-50** is a 50-layer deep convolutional neural network optimized for **Texas Instruments MPU devices**. This folder provides two production-ready ONNX variants that deliver industry-leading accuracy on ImageNet classification while maintaining efficient computation suitable for edge deployment.

This folder contains two distinct architectural generations of ResNet-50:

- **resNet50** — ResNet-50 **v1.5**, the modern de-facto standard. Used by PyTorch (`torchvision`), TensorFlow, and most current frameworks. The stride-2 downsampling in each bottleneck block is applied in the **3×3 convolution** rather than the 1×1, which improves accuracy with no added parameters. This is the version most practitioners encounter today.

- **resnet50-v1** — ResNet-50 **v1**, the original architecture from He et al. (2016) as published in the ONNX Model Zoo (opset 7). Stride-2 is applied in the **1×1 convolution**. Useful when strict reproducibility with the original paper or ONNX Model Zoo benchmarks is required.

The two variants differ only in where the stride-2 downsampling is placed inside each bottleneck block: moving the stride to the 3×3 conv (v1.5) preserves more spatial information before downsampling, which accounts for the ~1.2% accuracy gain over v1 at zero extra cost in parameters or FLOPs.

> **Which should I use?** For new projects, prefer **resNet50 (v1.5)** — it is more accurate and is the implementation underlying most pre-trained weights available today. Use **resnet50-v1** when you need exact compatibility with the original ONNX Model Zoo model or are comparing against v1 benchmarks.

---

## Model Variants

| Model | Architecture | Params | Top-1 Accuracy | Validated Devices | Config |
|-------|--------------|--------|-----------------|--------------------|--------|
| `resNet50` | ResNet-50 v1.5 (stride-2 in 3×3 conv) | ~25.6M | 76.15% | TDA4VH | [resnet50_config.yaml](resnet50_config.yaml) |
| `resnet50-v1` | ResNet-50 v1, original (stride-2 in 1×1 conv) | ~25.6M | 74.93% | TDA4VH | [resnet50-v1_config.yaml](resnet50-v1_config.yaml) |

**Recommended for edge deployment:** `resNet50` (v1.5) — highest accuracy with the same compute cost (4.1 GigaMACs) as the original v1.

---

## Quick Start

### Prerequisites

```bash
pip install onnx>=1.22.0 onnxruntime>=1.23.2
```

### Export the Model

```bash
# Download and prepare the default model (resNet50, v1.5)
python prepare_model.py

# Download and prepare a specific variant via its .link file
python prepare_model.py --link-file resnet50-v1.onnx.link

# Skip download and only fix shapes on an already-downloaded model
python prepare_model.py --link-file resnet50.onnx.link --skip-download

# Use a custom input resolution
python prepare_model.py --link-file resnet50.onnx.link --height 256 --width 256
```

The script automatically:
- Parses the `.link` file to get the download URL and output filename
- Downloads the ONNX model from HuggingFace (unless `--skip-download` is set)
- Fixes dynamic input shapes to a static shape (default `[1, 3, 224, 224]`)
- Runs ONNX shape inference and optional `onnx-simplifier` optimization
- Validates the resulting model and confirms all shapes are fixed

### Compile and Infer uing TIDL Runner

**Compile using TIDL Runner - on PC**

```bash
tidlrunner-cli compile --target_device J784S4 \
  --config_path resnet50_config.yaml
```

**Run Inference Benchmark - on device**

```bash
tidlrunner-cli infer --target_device J784S4 \
  --config_path resnet50_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision 
            and pattern recognition},
  pages={770--778},
  year={2016}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| **Source (resNet50)** | [onnx-community/resnet-50-ONNX](https://huggingface.co/onnx-community/resnet-50-ONNX) |
| **Source (resnet50-v1)** | [onnxmodelzoo/resnet50-v1-7](https://huggingface.co/onnxmodelzoo/resnet50-v1-7) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

<table>
<tr>
<td align="center">

**MobileNetV3**
Mobile-optimized CNN
Lighter alternative

</td>
<td align="center">

**ConvNeXt**
Modern CNN successor
Higher accuracy

</td>
<td align="center">

**DINO (ResNet-50)**
Self-supervised ResNet
No-label pre-training

</td>
<td align="center">

**ViT**
Vision Transformer
Attention-based backbone

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
