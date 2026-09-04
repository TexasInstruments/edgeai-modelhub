---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# RT-DETRv2 for TI EdgeAI

### Real-Time End-to-End Detection Transformer, v2

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**RT-DETRv2** (Real-Time Detection Transformer v2) is the improved version of RT-DETR, presented at **CVPR 2024**. It is a real-time, end-to-end object detection transformer that eliminates the need for hand-crafted anchor boxes and NMS post-processing. Built on a **ResNet-vd hybrid encoder** backbone with a transformer decoder, it achieves state-of-the-art accuracy-speed trade-offs on COCO across five size variants (S, M*, M, L, X).

Each exported model produces two outputs (batch=1 by default): `pred_boxes` `[1, 300, 4]` (CxCyWH normalised to [0,1], per-query box predictions) and `pred_logits` `[1, 300, 80]` (raw class logits — apply sigmoid for probabilities). Boxes are relative to the input image size.

All variants are released under the Apache 2.0 license.

---

## Model Variants

| Model | Backbone | Params (M) | FLOPs (G) | mAP[.5:.95]% | mAP[.50]% | Validated Devices | Config |
|-------|----------|-----------|-----------|--------------|-----------|--------------------|--------|
| `rtdetrv2_s` | ResNet-18vd | 20 | 60 | 48.1 | 65.1 | TDA4VH | [rtdetrv2_s_config.yaml](rtdetrv2_s_config.yaml) |
| `rtdetrv2_ms` | ResNet-34vd | 31 | 92 | 49.9 | 67.5 | TDA4VH | [rtdetrv2_ms_config.yaml](rtdetrv2_ms_config.yaml) |
| `rtdetrv2_m` | ResNet-50vd-m | 36 | 100 | 51.9 | 69.9 | TDA4VH | [rtdetrv2_m_config.yaml](rtdetrv2_m_config.yaml) |
| `rtdetrv2_l` | ResNet-50vd | 42 | 136 | 53.4 | 71.6 | TDA4VH | [rtdetrv2_l_config.yaml](rtdetrv2_l_config.yaml) |
| `rtdetrv2_x` | ResNet-101vd | 76 | 259 | 54.3 | 72.8 | TDA4VH | [rtdetrv2_x_config.yaml](rtdetrv2_x_config.yaml) |

> mAP evaluated on COCO val2017. Input resolution 640x640 for all variants.

**Recommended for edge deployment:** `rtdetrv2_s` (best accuracy/compute trade-off, smallest variant)

---

## Quick Start

### Prerequisites

```bash
# Core dependencies (auto-installed by prepare_model.py)
pip install torch>=2.0.1 torchvision>=0.15.2 scipy PyYAML onnx

# Optional: ONNX simplifier
pip install onnxsim

# Inference and deployment
pip install onnxruntime>=1.16.0
```

### Export the Model

```bash
# List all available variants with accuracy and latency info
python prepare_model.py --list-models

# Export the default model (rtdetrv2_s)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model rtdetrv2_m

# Export multiple variants at once
python prepare_model.py --model rtdetrv2_s rtdetrv2_m rtdetrv2_l

# Export with a custom input resolution
python prepare_model.py --model rtdetrv2_l --shape 800 800

# Export from a locally trained checkpoint
python prepare_model.py --model rtdetrv2_m --weights /path/to/custom.pth

# Export with ONNX simplification applied
python prepare_model.py --model rtdetrv2_s --simplify
```

The script automatically:
- Clones the RT-DETR source repository (to `~/.cache/rtdetr_src`) on first use
- Downloads pretrained COCO weights from GitHub Releases on first use
- Builds the deploy-mode model (drops training-only components) and wraps it to return `pred_boxes` and `pred_logits`
- Exports to ONNX (opset 16 by default) and runs ONNX shape inference
- Optionally applies `onnxsim` simplification when `--simplify` is passed

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/rtdetrv2_s_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/rtdetrv2_s_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use RT-DETRv2 in your research, please cite:

```bibtex
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
  title     = {RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
  author    = {Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
  year      = {2024},
  eprint    = {2407.17140},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2407.17140}
}

@misc{lv2023detrs,
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  author    = {Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei
               and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
  year      = {2023},
  eprint    = {2304.08069},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2407.17140](https://arxiv.org/abs/2407.17140) |
| **Original RT-DETR Paper** | [arXiv:2304.08069](https://arxiv.org/abs/2304.08069) |
| **Source Code** | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**RF-DETR**
Real-time DETR variant
Open-vocabulary friendly design

</td>
<td align="center">

**DEIMv2**
Improved DETR training recipe
Faster convergence, strong accuracy

</td>
<td align="center">

**Deformable-DETR**
Deformable attention DETR
Better small-object detection

</td>
<td align="center">

**DETR**
Original detection transformer
Foundation of the DETR family

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
