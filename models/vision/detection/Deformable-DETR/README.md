---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# Deformable DETR for TI EdgeAI

### Deformable Attention for Fast-Converging Transformer Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org)

</div>

---

## Overview

**Deformable DETR** (Deformable Transformers for End-to-End Object Detection) is a transformer-based detector from SenseTime / fundamentalvision that addresses the slow convergence and limited feature resolution of the original DETR. Its key innovation is a **deformable attention module** that attends to only a small set of key sampling points around a reference point rather than all feature map positions, reducing complexity from **O(H²W²) to O(HW)**.

This efficient attention mechanism makes it practical to use **multi-scale feature maps**, which improves detection accuracy — especially on small objects — while training in **10× fewer epochs** than DETR. Five variants are provided, ranging from a lightweight single-scale model to a two-stage design with iterative bounding box refinement.

Deformable DETR uses **300 query slots** (vs. 100 in DETR) and **sigmoid focal loss** for classification (no explicit background class); post-processing applies a score threshold rather than softmax + background filtering.

---

## Model Variants

| Model | Params | FLOPs | mAP[.5:.95]% | Validated Devices | Config |
|-------|--------|-------|--------------|--------------------|--------|
| `deformable_detr_single_scale` | 34M | 78G | 39.4 | TDA4VH | [deformable_detr_single_scale_config.yaml](deformable_detr_single_scale_config.yaml) |
| `deformable_detr_single_scale_dc5` | 34M | 128G | 41.5 | N/A | N/A |
| `deformable_detr` | 40M | 173G | 44.5 | N/A | N/A |
| `deformable_detr_plus_iterative_bbox_refinement` | 41M | 173G | 46.2 | N/A | N/A |
| `deformable_detr_two_stage` | 41M | 173G | 46.9 | N/A | N/A |

**Recommended for edge deployment:** `deformable_detr` (multi-scale, best accuracy/compute trade-off)

> mAP values on COCO val2017. All variants use a ResNet-50 backbone pretrained on ImageNet, 800×800 input.
> The DC5 variant is disabled for TIDL deployment — TIDL does not support dilated convolutions in ResNet; its `.onnx` is provided for reference only.
> Only `deformable_detr_single_scale` currently ships with a validated TIDL config; the remaining variants have no `*_config.yaml` in this folder.

---

## Quick Start

### Prerequisites

```bash
# Core dependencies (auto-installed by prepare_model.py if missing)
pip install torch>=1.12.0 torchvision>=0.13.0 onnx>=1.14.0 scipy gdown>=5.2.0

# ONNX inference
pip install onnxruntime>=1.15.0
```

### Export the Model

```bash
# List all available variants with accuracy and parameter info
python prepare_model.py --list-models

# Export the default model (deformable_detr - multi-scale, recommended)
python prepare_model.py

# Export a specific variant
python prepare_model.py --model deformable_detr_single_scale

# Export all variants (skips any already exported)
python prepare_model.py --model all

# Use HuggingFace Hub instead of Google Drive (recommended on corporate networks)
python prepare_model.py --method optimum --model all
```

The script automatically:
- Installs missing dependencies (torch, torchvision, onnx, scipy, gdown) if not present
- Clones the [Deformable-DETR repository](https://github.com/fundamentalvision/Deformable-DETR) into `~/.cache/deformable_detr` (or downloads weights from HuggingFace Hub with `--method optimum`)
- Installs a pure-Python fallback for the multi-scale deformable attention module — no CUDA compilation required
- Downloads pretrained COCO weights and builds the model with the correct architecture flags
- Exports to ONNX (opset 17 by default), simplifies the graph with onnx-simplifier, and fixes float64 nodes for TIDL compatibility
- Validates the exported graph and saves it as `<model_key>.onnx`

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/deformable_detr_single_scale_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/deformable_detr_single_scale_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

```bibtex
@article{zhu2020deformable,
  title   = {Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author  = {Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and
             Wang, Xiaogang and Dai, Jifeng},
  journal = {arXiv preprint arXiv:2010.04159},
  year    = {2020}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2010.04159](https://arxiv.org/abs/2010.04159) |
| **Source Code** | [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) |
| **Dataset** | [COCO](https://cocodataset.org) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

<table>
<tr>
<td align="center">

**DETR**
Original transformer detector
Predecessor to Deformable DETR

</td>
<td align="center">

**RT-DETRv2**
Real-time transformer detector
Modern DETR-style architecture

</td>
<td align="center">

**RF-DETR**
Receptive-field enhanced DETR
Recent DETR-family variant

</td>
<td align="center">

**DEIMv2**
Improved DETR training recipe
Faster convergence, higher accuracy

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
