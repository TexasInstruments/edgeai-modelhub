---
license: agpl-3.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# YOLO26 for TI EdgeAI

### Native End-to-End Object Detector for Real-Time Edge Deployment

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge)](https://opensource.org/licenses/AGPL-3.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**YOLO26** is the newest generation of the Ultralytics YOLO family, released in January 2026. Its detection head is natively end-to-end: by default it predicts final boxes directly, without a separate non-maximum suppression (NMS) post-processing step, which simplifies deployment and reduces post-processing latency. The head also removes Distribution Focal Loss (DFL) from box regression, lowering head complexity while keeping an unconstrained regression range.

The training recipe pairs these architectural changes with **MuSGD** (a hybrid Muon + SGD optimizer), **Progressive Loss** (which shifts supervision emphasis toward the inference-time head), and **STAL**, a Small-Target-Aware Label Assignment scheme that preserves positive label coverage for small objects. Together these updates improve the accuracy/latency trade-off over YOLO11 across all five model scales and give YOLO26n notably faster CPU ONNX inference, making the family well suited to power- and latency-constrained edge deployments.

These ONNX models cover the five COCO-pretrained detection scales (n/s/m/l/x, 80 classes), exported and shape-fixed to a static `640×640` input for TIDL compilation on TI edge SoCs.

> See [YOLO11](../YOLO11/) for the previous-generation, NMS-based YOLO models.

---

## Model Variants

| Model | Input Size | mAP[.5:.95]% | Validated Devices | Config |
|-------|-----------|--------------|--------------------|--------|
| `yolo26n` | 640×640 | 40.9 | TDA4VH | [yolo26n_model_config.yaml](yolo26n_model_config.yaml) |
| `yolo26s` | 640×640 | 48.6 | TDA4VH | [yolo26s_model_config.yaml](yolo26s_model_config.yaml) |
| `yolo26m` | 640×640 | 53.1 | TDA4VH | [yolo26m_model_config.yaml](yolo26m_model_config.yaml) |
| `yolo26l` | 640×640 | 55.0 | TDA4VH | [yolo26l_model_config.yaml](yolo26l_model_config.yaml) |
| `yolo26x` | 640×640 | 57.5 | TDA4VH | [yolo26x_model_config.yaml](yolo26x_model_config.yaml) |

**Recommended for edge deployment:** `yolo26n` (best accuracy/compute trade-off)

---

## Quick Start

### Prerequisites

```bash
pip install onnx>=1.22.0 onnxruntime>=1.23.2
```

### Export the Model

```bash
# Prepare the default model (yolo26n)
python prepare_model.py

# Prepare a specific model variant
python prepare_model.py --model yolo26s

# Prepare multiple variants in one run
python prepare_model.py --model yolo26n yolo26s yolo26m

# Prepare every supported variant
python prepare_model.py --model all

# List all supported variants and their local download/conversion status
python prepare_model.py --list-models

# Re-run shape fixing on an already-downloaded ONNX
python prepare_model.py --model yolo26n --skip-download
```

The script automatically:
- Parses the variant's `.link` file to get the HuggingFace download URL
- Downloads the model with `curl` if it isn't already present locally
- Fixes dynamic input dimensions to a static shape (default `[1, 3, 640, 640]`)
- Runs ONNX shape inference and optional `onnx-simplifier` optimization
- Validates the resulting ONNX model structure

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/yolo26n_model_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/yolo26n_model_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use these models, please cite:

```bibtex
@article{jocher2026yolo26,
  title={Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author={Jocher, Glenn and Qiu, Jing and Liu, Mengyu and Lyu, Shuai and
          Akyon, Fatih Cagatay and Kalfaoglu, Muhammet Esat},
  journal={arXiv preprint arXiv:2606.03748},
  year={2026}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2606.03748](https://arxiv.org/abs/2606.03748) |
| **Source Code** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **Model Docs** | [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**YOLO11**
Predecessor generation
NMS-based detection

</td>
<td align="center">

**YOLOv8**
Earlier YOLO generation
Widely adopted baseline

</td>
<td align="center">

**YOLOX**
Anchor-free detector
Decoupled head design

</td>
<td align="center">

**RT-DETRv2**
Transformer-based detector
Real-time DETR variant

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
