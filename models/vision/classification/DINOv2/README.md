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

### Self-Supervised Vision Transformer Backbone for Image Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## Overview

**DINOv2** (Self-**Di**stillation with **No** Labels v2) is a self-supervised Vision Transformer pre-training method from Meta AI. It produces high-performance visual features using a purely self-supervised training regime on 142M images — no labels required during pre-training. The pretrained backbones are paired with a lightweight linear classification head for ImageNet-1K inference, outputting 1000-class logits.

All models use a **ViT/14** patch size (14x14 patches) and are evaluated at **224x224** input resolution. Some variants add **register tokens** ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588)) — extra learnable tokens that absorb the attention artifacts otherwise seen in patch-token feature maps, giving slightly better accuracy with the same backbone size.

> See [DINO](../DINO/) for the original first-generation models.

---

## Model Variants

| Model | Architecture | Params | GFLOPs | Top-1 Acc | Validated Devices | Config |
|-------|-------------|--------|--------|-----------|----------|--------|
| `dinov2_vits14_lc` | ViT-S/14 distilled | 21M | 4.6 | 81.1% | TDA4VH | [dinov2_vits14_lc_config.yaml](dinov2_vits14_lc_config.yaml) |
| `dinov2_vits14_reg_lc` | ViT-S/14 distilled + registers | 21M | 4.6 | 80.9% | TDA4VH | [dinov2_vits14_reg_lc_config.yaml](dinov2_vits14_reg_lc_config.yaml) |
| `dinov2_vitb14_lc` | ViT-B/14 distilled | 86M | 17.6 | 84.5% | TDA4VH | [dinov2_vitb14_lc_config.yaml](dinov2_vitb14_lc_config.yaml) |
| `dinov2_vitb14_reg_lc` | ViT-B/14 distilled + registers | 86M | 17.6 | 84.6% | TDA4VH | [dinov2_vitb14_reg_lc_config.yaml](dinov2_vitb14_reg_lc_config.yaml) |

**Recommended for edge deployment:** `dinov2_vits14_lc` (best accuracy/compute trade-off)

> **Note:** ViT-L/14 and ViT-g/14 variants (`dinov2_vitl14_lc`, `dinov2_vitg14_lc`, and their `_reg_lc` counterparts) are excluded from this repo — they require ~16 GB+ RAM to export and are not suitable for edge (TIDL) deployment.

---

## Quick Start

### Prerequisites

```bash
pip install torch torchvision onnx>=1.22.0 onnxruntime>=1.23.2 onnx-simplifier
```

### Export the Model

```bash
# Export the default model (ViT-S/14 distilled)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model dinov2_vitb14_lc

# Export the registers variant
python prepare_model.py --model dinov2_vits14_reg_lc

# Export all edge-suitable models
python prepare_model.py --model all

# Re-run shape fixing on an already-exported ONNX
python prepare_model.py --model dinov2_vits14_lc --skip-export
```

The script automatically:
- Loads the pretrained backbone + linear classification head from PyTorch Hub (`facebookresearch/dinov2`)
- Exports to ONNX (opset 17) with a dynamic batch axis
- Fixes input shapes to `[1, 3, 224, 224]` and propagates shapes via ONNX shape inference
- Runs onnx-simplifier (`onnxsim`) and validates the final model with `onnx.checker`

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/dinov2_vits14_lc_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/dinov2_vits14_lc_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use these models, please cite:

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

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| **Registers Paper** | [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) |
| **Source Code** | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **DINO** | [Predecessor model](../DINO/) |

---

## Related Models

<table>
<tr>
<td align="center">

**DINO**
Predecessor
First-generation self-supervised ViT

</td>
<td align="center">

**ViT**
Alternative backbone
Supervised transformer

</td>
<td align="center">

**ResNet**
CNN backbone
Lower compute

</td>
<td align="center">

**ConvNeXt**
Modern CNN
Transformer-inspired design

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
