---
license: apache-2.0
tags:
- vision
- image-classification
- self-supervised
datasets:
- imagenet-1k
---

<div align="center">

# DINO for TI EdgeAI

### Self-Supervised Vision Transformer Backbone for Image Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**DINO** (Self-**Di**stillation with **No** labels) is a self-supervised Vision Transformer pre-training method from Meta AI. The backbone models produce rich feature embeddings that achieve strong performance on ImageNet classification without any labels during pre-training.

These ONNX models include the **full backbone + pretrained linear classification head**, outputting 1000-class ImageNet logits `[1, 1000]`. Feature extraction follows DINO's `eval_linear.py` conventions:
- **ViT-S models**: CLS tokens from last 4 blocks concatenated → `[B, 1536]`
- **ViT-B models**: CLS token + averaged patch tokens (interleaved) → `[B, 1536]`
- **ResNet-50**: avgpool output → `[B, 2048]`

> See [DINOv2](../DINOv2/) for the improved second-generation models.

---

## Model Variants

| Model | Architecture | Params | Linear Top-1 | k-NN Top-1 | Validated Devices | Config |
|-------|-------------|--------|-------------|-----------|----------|--------|
| `dino_vits16` | ViT-S/16 | 21M | 77.0% | 74.5% | TDA4VH | [dino_vits16_config.yaml](dino_vits16_config.yaml) |
| `dino_vits8`  | ViT-S/8  | 21M | 79.7% | 78.3% | TDA4VH | [dino_vits8_config.yaml](dino_vits8_config.yaml) |
| `dino_vitb16` | ViT-B/16 | 85M | 78.2% | 76.1% | TDA4VH | [dino_vitb16_config.yaml](dino_vitb16_config.yaml) |
| `dino_vitb8`  | ViT-B/8  | 85M | 80.1% | 77.4% | TDA4VH | [dino_vitb8_config.yaml](dino_vitb8_config.yaml) |
| `dino_resnet50` | ResNet-50 | 23M | 75.3% | 67.5% | TDA4VH | [dino_resnet50_config.yaml](dino_resnet50_config.yaml) |

**Recommended for edge deployment:** `dino_vits16` (best accuracy/compute trade-off)

---

## Quick Start

### Prerequisites

```bash
pip install onnx>=1.22.0 onnxruntime>=1.23.2
```

### Export the Model

```bash
# Export the default model (ViT-S/16)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model dino_vitb16

# Export all supported models
python prepare_model.py --model all

# Re-run shape fixing on an already-exported ONNX
python prepare_model.py --model dino_vits16 --skip-export
```

The script automatically:
- Loads pretrained backbone from PyTorch Hub (`facebookresearch/dino:main`)
- Downloads pretrained linear classification weights from Meta AI
- Combines backbone + linear head into a single classification model
- Exports to ONNX (opset 17) and fixes input shapes to [1, 3, 224, 224]
- Validates the model outputs `[1, 1000]` class logits

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/dino_vits16_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/dino_vits16_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use these models, please cite:

```bibtex
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and
          J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr
          and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF International Conference
             on Computer Vision (ICCV)},
  year={2021}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) |
| **Source Code** | [facebookresearch/dino](https://github.com/facebookresearch/dino) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **DINOv2** | [Improved successor](../DINOv2/) |

---

## Related Models

<table>
<tr>
<td align="center">

**DINOv2**
Improved DINO
Higher accuracy

</td>
<td align="center">

**ViT-S/16**
Recommended
Best edge trade-off

</td>
<td align="center">

**ResNet-50**
CNN backbone
Lower compute

</td>
<td align="center">

**CLIP**
Vision-Language
Zero-shot capable

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
