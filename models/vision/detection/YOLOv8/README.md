---
license: agpl-3.0
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# YOLOv8 for TI EdgeAI

### Real-Time Anchor-Free Object Detector

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge)](https://opensource.org/licenses/AGPL-3.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**YOLOv8** is Ultralytics' real-time object detector, building on the advancements of previous YOLO versions with an **anchor-free split Ultralytics head** design that improves accuracy and speeds up the detection process compared to earlier anchor-based approaches. It combines a state-of-the-art backbone and neck architecture for improved feature extraction with an optimized accuracy-speed trade-off, making it suitable for real-time detection across a wide range of applications.

YOLOv8 is offered in five size variants — n, s, m, l, x — spanning a wide accuracy-speed trade-off, from resource-constrained edge devices up to high-throughput deployments. This model is optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, targeting edge computer vision use cases such as industrial automation, smart cameras, robotics, and IoT vision.

---

## Model Variants

| Model | Params (M) | Input Size | mAP[.5:.95]% | Validated Devices | Config |
|-------|------------|------------|--------------|--------------------|--------|
| `yolov8n` | 3.2 | 640×640 | 37.3 | TDA4VH | [yolov8n_config.yaml](yolov8n_config.yaml) |
| `yolov8s` | 11.2 | 640×640 | 44.9 | N/A | N/A |
| `yolov8m` | 25.9 | 640×640 | 50.2 | TDA4VH | [yolov8m_config.yaml](yolov8m_config.yaml) |
| `yolov8l` | 43.7 | 640×640 | 52.9 | N/A | N/A |
| `yolov8x` | 68.2 | 640×640 | 53.9 | N/A | N/A |

**Recommended for edge deployment:** `yolov8n` (best accuracy/compute trade-off, smallest footprint)

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
# Export the default model (yolov8n)
python prepare_model.py

# Export specific variants
python prepare_model.py --models yolov8n yolov8m

# Export all variants
python prepare_model.py --models all

# List all supported variants
python prepare_model.py --list-models

# Export with a custom output directory or export format
python prepare_model.py --models yolov8n --output-dir ./exports --format onnx
```

The script automatically:
- Installs required runtime dependencies (`onnx`, `ultralytics`, `onnxslim`, `onnxruntime`)
- Loads the requested Ultralytics YOLOv8 checkpoint (`.pt`), downloading it on first use
- Exports the model to ONNX (opset 19 by default, configurable via `--opset`)
- Supports alternate export formats (`torchscript`, `tflite`, `pb`, `saved_model`, `coreml`, and more) via `--format`

### Compile and Infer uing TIDL Runner

**Compile using TIDL Runner - on PC**

```bash
tidlrunner-cli compile --target_device J784S4 \
  --config_path yolov8n_config.yaml
```

**Run Inference Benchmark - on device**

```bash
tidlrunner-cli infer --target_device J784S4 \
  --config_path yolov8n_config.yaml
```

To evaluate accuracy instead, replace `infer` with `evaluate` in the command above.

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

Ultralytics has not published a formal research paper for YOLOv8 due to the rapidly evolving nature of the models. If you use the YOLOv8 model or any other software from the Ultralytics repository in your work, please cite it using the following format:

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Source Code** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **Documentation** | [YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **EdgeAI MPU Overview** | [GitHub](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu) |
| **TI EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**YOLO11**
Newer Ultralytics generation
Improved accuracy/speed

</td>
<td align="center">

**YOLO26**
Latest Ultralytics generation
Unified, end-to-end detection

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
