---
license: apache-2.0
tags:
- vision
- image-detection
- image-segmentation
datasets:
- COCO
---

<div align="center">

# RF-DETR for TI EdgeAI

### Real-Time Transformer Detection with a DINOv2 Backbone

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org/)

</div>

---

## Overview

**RF-DETR** is a real-time transformer-based object detection (and instance segmentation) architecture developed by Roboflow, achieving state-of-the-art accuracy/latency trade-offs on COCO (presented at ICLR 2026). It builds on a **DINOv2 ViT backbone** paired with a DETR-style decoder and a hierarchical feature pyramid, giving it strong small-object and dense-scene performance without the NMS and anchor-tuning overhead of traditional detectors.

RF-DETR ships in six size variants, Nano through 2XLarge, for flexible accuracy-speed trade-offs. The Nano through Large variants are released under Apache 2.0; XLarge and 2XLarge require the `rfdetr[plus]` extra and are licensed under PML 1.0. This folder packages the ONNX exports and TIDL configs for the Apache-licensed detection variants (Nano/Small/Medium/Large); the `prepare_model.py` script can additionally export the PML-licensed XLarge/2XLarge detection variants and the segmentation family on request.

---

## Model Variants

| Model | Input Size | mAP[.5:.95]% | mAP[.50]% | Validated Devices | Config |
|-------|-----------|--------------|-----------|--------------------|--------|
| `rfdetr_nano` | 384×384 | 48.4 | 67.6 | TDA4VH | [rfdetr_nano_config.yaml](rfdetr_nano_config.yaml) |
| `rfdetr_small` | 512×512 | 53.0 | 72.1 | TDA4VH | [rfdetr_small_config.yaml](rfdetr_small_config.yaml) |
| `rfdetr_medium` | 576×576 | 54.7 | 73.6 | TDA4VH | [rfdetr_medium_config.yaml](rfdetr_medium_config.yaml) |
| `rfdetr_large` | 704×704 | 56.5 | 75.1 | TDA4VH | [rfdetr_large_config.yaml](rfdetr_large_config.yaml) |

> mAP values are on COCO val2017. `rfdetr_xlarge` (700×700, mAP[.5:.95] 58.6) and `rfdetr_2xlarge` (880×880, mAP[.5:.95] 60.1) are available via `prepare_model.py --plus` but are licensed under PML 1.0 and are not shipped as ONNX/config files in this folder.

**Recommended for edge deployment:** `rfdetr_nano` (best accuracy/compute trade-off, smallest)

---

## Quick Start

### Prerequisites

```bash
# ONNX export dependencies
pip install "rfdetr[onnx]"

# For XLarge / 2XLarge variants (PML 1.0 license)
pip install "rfdetr[onnx,plus]"

# Inference and deployment
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
```

### Export the Model

```bash
# List all available variants with accuracy and latency info
python prepare_model.py --list-models

# Export the default model (rfdetr_nano)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model rfdetr_medium

# Export multiple variants at once
python prepare_model.py --model rfdetr_nano rfdetr_small rfdetr_medium rfdetr_large

# Export XLarge / 2XLarge (requires rfdetr[plus], PML 1.0 license)
python prepare_model.py --model rfdetr_xlarge rfdetr_2xlarge --plus
```

The script automatically:
- Installs `rfdetr[onnx]` (or `rfdetr[onnx,plus]` for XLarge/2XLarge) if not already present
- Downloads pretrained COCO weights from HuggingFace on first use
- Exports the selected variant(s) to ONNX (opset 17 by default, static batch dimension)
- Saves the result as `rfdetr_<variant>.onnx` in the output directory

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/rfdetr_nano_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/rfdetr_nano_config.yaml
```

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use RF-DETR in your research, please cite:

```bibtex
@software{rfdetr2025,
  title        = {RF-DETR},
  author       = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  year         = {2025},
  publisher    = {Roboflow},
  url          = {https://github.com/roboflow/rf-detr},
  note         = {International Conference on Learning Representations (ICLR) 2026}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **RF-DETR Source Code** | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) |
| **RF-DETR Documentation** | [rfdetr.roboflow.com](https://rfdetr.roboflow.com) |
| **RF-DETR on HuggingFace** | [huggingface.co/roboflow](https://huggingface.co/roboflow) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

<table>
<tr>
<td align="center">

**RT-DETRv2**
Real-time DETR variant
Anchor-free, NMS-free

</td>
<td align="center">

**DEIMv2**
DETR-family detector
Improved matching/training

</td>
<td align="center">

**Deformable-DETR**
Sparse attention DETR
Faster convergence

</td>
<td align="center">

**DETR**
Original transformer detector
End-to-end set prediction

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
