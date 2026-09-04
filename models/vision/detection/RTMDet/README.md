---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# RTMDet for TI EdgeAI

### Real-Time Object Detector with a CSPNeXt Backbone

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**RTMDet** is a high-performance real-time object detector from OpenMMLab with a CSPNeXt backbone and an efficient anchor-free detection head. It achieves excellent accuracy-speed trade-offs across five model sizes (tiny, s, m, l, x), making it suitable for a wide range of deployment scenarios from resource-constrained edge devices to high-throughput server deployments.

This RTMDet model is optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready object detection with minimal setup.

---

## Model Variants

| Model | Input Size | mAP[.5:.95]% | Validated Devices | Config |
|-------|-----------|--------------|--------------------|--------|
| `rtmdet_tiny` | 640x640 | 40.9 | TDA4VH | [rtmdet_tiny_config.yaml](rtmdet_tiny_config.yaml) |
| `rtmdet_s` | 640x640 | 44.5 | TDA4VH | [rtmdet_s_config.yaml](rtmdet_s_config.yaml) |
| `rtmdet_m` | 640x640 | 49.3 | TDA4VH | [rtmdet_m_config.yaml](rtmdet_m_config.yaml) |
| `rtmdet_l` | 640x640 | 51.4 | TDA4VH | [rtmdet_l_config.yaml](rtmdet_l_config.yaml) |
| `rtmdet_x` | 640x640 | 52.8 | TDA4VH | [rtmdet_x_config.yaml](rtmdet_x_config.yaml) |

**Recommended for edge deployment:** `rtmdet_tiny` (smallest, best accuracy/compute trade-off)

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
# Export all variants (default)
python prepare_model.py

# Export specific variants
python prepare_model.py --models tiny
python prepare_model.py --models tiny s m

# Export without ONNX simplification
python prepare_model.py --models tiny --no-simplify

# Force regeneration of .link files
python prepare_model.py --generate-links
```

The script automatically:
- Installs `mmcv-lite` and `mmdet` (and other required dependencies)
- Downloads the PyTorch checkpoint referenced by each variant's `.onnx.link` file from OpenMMLab
- Downloads the matching mmdetection config files (pinned to tag `v3.3.0`)
- Builds the model with `mmdet.apis.init_detector` and wraps it to emit decoded `boxes` (xyxy) and per-class sigmoid `scores` (NMS is left for on-device post-processing)
- Exports to ONNX (opset 13), fixes the batch dimension to 1, and re-runs shape inference
- Optionally simplifies the model using `onnx-simplifier`

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/rtmdet_tiny_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/rtmdet_tiny_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use RTMDet in your research, please cite:

```bibtex
@article{lyu2022rtmdet,
  title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
  author={Lyu, Chengqi and Zhang, Wenwei and Huang, Haian and Zhou, Yue and Wang, Yudong and Liu, Yanyi and Zhang, Shilong and Chen, Kai},
  journal={arXiv preprint arXiv:2212.07784},
  year={2022}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2212.07784](https://arxiv.org/abs/2212.07784) |
| **Source Code** | [open-mmlab/mmdetection (rtmdet configs)](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**YOLOX**
Anchor-free CNN detector
Similar single-stage design

</td>
<td align="center">

**YOLOv8**
CNN-based real-time detector
Comparable accuracy/speed range

</td>
<td align="center">

**YOLO11**
Latest Ultralytics YOLO
Improved efficiency

</td>
<td align="center">

**RT-DETRv2**
Real-time transformer detector
NMS-free alternative

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
