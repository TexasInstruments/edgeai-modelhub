---
license: agpl-3.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# YOLO11 for TI EdgeAI

### Real-Time Object Detector with an Enhanced Backbone and C2PSA Attention

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge)](https://opensource.org/licenses/AGPL-3.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**YOLO11** is Ultralytics' successor to YOLOv8, released in September 2024. It keeps the overall one-stage, anchor-free detection pipeline but reworks the backbone and neck for more efficient feature extraction: the `C2f` block used throughout YOLOv8 is replaced by a `C3k2` block (a faster variant of the CSP bottleneck that can switch between smaller convolution kernels for efficiency), and a `C2PSA` (Cross-Stage Partial with Spatial Attention) module is inserted after the backbone's SPPF layer to add lightweight spatial-attention refinement to the extracted features. Together with a refined training pipeline, these changes let YOLO11 reach higher COCO mAP than the equivalent YOLOv8 scale while using noticeably fewer parameters — for example, YOLO11m matches or beats YOLOv8m's accuracy with about 22% fewer parameters.

YOLO11 is offered in five size variants — n, s, m, l, x — spanning a wide accuracy-speed trade-off, from resource-constrained edge devices up to high-throughput deployments. This model is optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, targeting edge computer vision use cases such as industrial automation, smart cameras, robotics, and IoT vision.

> See [YOLO26](../YOLO26/) for the newest Ultralytics generation, with native end-to-end (NMS-free) detection.

---

## Model Variants

| Model | Params (M) | Input Size | mAP[.5:.95]% | Validated Devices | Config |
|-------|------------|------------|--------------|--------------------|--------|
| `yolo11n` | 2.6 | 640×640 | 39.5 | TDA4VH, TDA4VL | [yolo11n_model_config.yaml](yolo11n_model_config.yaml) |
| `yolo11s` | 9.4 | 640×640 | 47.0 | TDA4VH, TDA4VL | [yolo11s_model_config.yaml](yolo11s_model_config.yaml) |
| `yolo11m` | 20.1 | 640×640 | 51.5 | TDA4VH, TDA4VL | [yolo11m_model_config.yaml](yolo11m_model_config.yaml) |
| `yolo11l` | 25.3 | 640×640 | 53.4 | TDA4VH, TDA4VL | [yolo11l_model_config.yaml](yolo11l_model_config.yaml) |
| `yolo11x` | 56.9 | 640×640 | 54.7 | TDA4VH, TDA4VL | [yolo11x_model_config.yaml](yolo11x_model_config.yaml) |

**Recommended for edge deployment:** `yolo11n` (best accuracy/compute trade-off, smallest footprint)

---

## Quick Start

### Prerequisites

```bash
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
pip install ultralytics
```

For TI hardware deployment, also set up **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)**.

If accessing this model from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Export the Model

```bash
# Prepare the default model (yolo11n)
python prepare_model.py

# Prepare a specific model variant
python prepare_model.py --model yolo11s

# Prepare multiple variants in one run
python prepare_model.py --model yolo11n yolo11s yolo11m

# Prepare every supported variant
python prepare_model.py --model all

# List all supported variants and their local download/conversion status
python prepare_model.py --list-models

# Re-run shape fixing on an already-downloaded ONNX
python prepare_model.py --model yolo11n --skip-download
```

The script automatically:
- Parses the variant's `.link` file to get the HuggingFace download URL for the `.pt` checkpoint
- Downloads the `.pt` model with `curl` if it isn't already present locally
- Converts the `.pt` model to ONNX (opset 17) using Ultralytics' `model.export()`
- Fixes dynamic input dimensions to a static shape (default `[1, 3, 640, 640]`)
- Runs ONNX shape inference and optional `onnx-simplifier` optimization, then validates the result

### Compile and Infer uing edgeai-tidlrunner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using edgeai-tidlrunner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/yolo11n_model_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/yolo11n_model_config.yaml
```

To evaluate accuracy instead, replace `infer` with `evaluate` in the command above.

### Compile and Infer using edgeai-tidl-tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

### Deploy using edgeai-tidl-tools:

Deplyment can be done using **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)**. For ONNX models, onnxruntime-tidl with TIDL acceleration can be used. Consult the documentation of edgeai-tidl-tools for more details.

---

## Citation

Ultralytics has not published a formal research paper for YOLO11 due to the rapidly evolving nature of the models. If you use the YOLO11 model or any other software from the Ultralytics repository in your work, please cite it using the following format:

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Source Code** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **Documentation** | [YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/) |
| **edgeai-tidl-tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **edgeai-tidlrunner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **EdgeAI MPU Overview** | [GitHub](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu) |
| **TI EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**YOLOv8**
Predecessor generation
Anchor-free split head

</td>
<td align="center">

**YOLO26**
Newest Ultralytics generation
NMS-free end-to-end detection

</td>
<td align="center">

**YOLOX**
Anchor-free YOLO variant
Decoupled head design

</td>
<td align="center">

**RTMDet**
CNN-based alternative
Real-time mmdetection model

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
