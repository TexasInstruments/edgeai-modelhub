---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# YOLOX for TI EdgeAI

### Anchor-Free YOLO with a Decoupled Head

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**YOLOX** is an anchor-free member of the YOLO family developed by Megvii, designed to close the gap between research and industrial object detection. Unlike earlier YOLO versions, YOLOX removes predefined anchor boxes and instead predicts objects directly, which simplifies the design and reduces the number of heuristic tuning parameters (e.g., anchor sizes) needed for a new dataset.

Two architectural changes distinguish YOLOX from anchor-based YOLO detectors: a **decoupled head** that separates classification and localization into independent branches (improving convergence and accuracy over the coupled head used in YOLOv3/v4/v5), and **SimOTA**, an advanced label-assignment strategy that formulates matching between predictions and ground-truth boxes as an optimal-transport problem to pick better positive samples during training.

This model is optimized for deployment on **Texas Instruments edge devices**, providing production-ready object detection for industrial automation, smart cameras, robotics, and IoT vision applications. It offers a range of variants — from the lightweight `yolox-nano` to the high-accuracy `yolox-x` — covering a wide accuracy/compute trade-off space.

---

## Model Variants

| Model | Model ID | Input Size | mAP[.5:.95]% | Validated Devices | Config |
|-------|----------|------------|--------------|--------------------|--------|
| `yolox_nano` | `od-mh8009` | 416×416 | 24.8 | TDA4VH | [yolox_nano_config.yaml](yolox_nano_config.yaml) |
| `yolox_tiny` | `od-mh8010` | 416×416 | 32.8 | TDA4VH | [yolox_tiny_config.yaml](yolox_tiny_config.yaml) |
| `yolox_s` | - | 640×640 | - | - | N/A |
| `yolox_m` | `od-mh8011` | 640×640 | 46.9 | TDA4VH | [yolox_m_config.yaml](yolox_m_config.yaml) |
| `yolox_l` | `od-mh8012` | 640×640 | 49.7 | TDA4VH | [yolox_l_config.yaml](yolox_l_config.yaml) |
| `yolox_x` | `od-mh8013` | 640×640 | 51.2 | TDA4VH | [yolox_x_config.yaml](yolox_x_config.yaml) |
| `yolox_darknet53` | `od-mh8014` | 640×640 | 47.4 | TDA4VH | [yolox_darknet53_config.yaml](yolox_darknet53_config.yaml) |

**Recommended for edge deployment:** `yolox_nano` (smallest, best accuracy/compute trade-off)

> `yolox_s` currently ships only as an ONNX export (`yolox_s.onnx`) with no TIDL config YAML available in this folder.

---

## Quick Start

### Prerequisites

```bash
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
pip install onnxsim  # For model simplification
```

### Export the Model

```bash
# Prepare a specific variant (downloads pre-built ONNX, or falls back to
# downloading the .pth checkpoint and converting it locally)
python prepare_model.py --model yolox_nano

# Prepare multiple variants at once
python prepare_model.py --model yolox_nano yolox_tiny yolox_m

# Prepare every supported variant
python prepare_model.py --model all

# Force re-download even if the ONNX/PTH file already exists locally
python prepare_model.py --model yolox_nano --force-download

# Prepare and verify accuracy on COCO val2017
python prepare_model.py --model yolox_nano --verify --num-val-images 500

# List all supported variants and their local download status
python prepare_model.py --list-models
```

The script automatically:
- Reads the source URL from each variant's `.onnx.link` file and downloads the pre-built ONNX from GitHub releases
- Falls back to downloading the PyTorch checkpoint from the `.pth.link` file and converting it to ONNX locally (via the official YOLOX `yolox.exp.get_exp` API) if the pre-built ONNX is unavailable
- Runs ONNX shape inference and hard-codes the batch dimension to 1
- Optionally simplifies the model using `onnx-simplifier`
- Optionally verifies accuracy on COCO val2017 using `pycocotools`, reporting mAP@[0.50:0.95] and mAP@0.50

Supported `--model` values: `yolox_nano`, `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x`, `yolox_darknet53`, or `all`.

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/yolox_nano_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/yolox_nano_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use YOLOX in your research, please cite:

```bibtex
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2107.08430](https://arxiv.org/abs/2107.08430) |
| **Source Code** | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**RTMDet**
Real-time single-stage detector
Distillation-enhanced backbone

</td>
<td align="center">

**YOLOv8**
Anchor-free single-stage detector
Improved training pipeline

</td>
<td align="center">

**YOLO11**
Latest Ultralytics YOLO
Refined efficiency/accuracy

</td>
<td align="center">

**RT-DETRv2**
Real-time DETR-based detector
Transformer decoder head

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
