---
license: other
license_name: deimv2-research-only
license_link: https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md
tags:
- vision
- image-detection
datasets:
- COCO
---

<div align="center">

# DEIMv2 for TI EdgeAI

### Dense One-to-One Matching Meets DINOv3 for Fast-Converging Detection

[![License](https://img.shields.io/badge/License-Non--Commercial-red?style=for-the-badge)](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org)

</div>

---

## Overview

**DEIMv2** is an evolution of the **DEIM** (DETR with Improved Matching) framework, extended with rich features from **DINOv3**. DEIM's core contribution — Dense One-to-One (Dense O2O) label assignment — accelerates convergence of DETR-style detectors versus the traditional sparse one-to-one matching used in DETR/Deformable-DETR, without sacrificing the end-to-end, NMS-free detection pipeline.

DEIMv2 spans eight model sizes from ultra-light (`Atto`) to extra-large (`X`), covering GPU, edge, and mobile deployment budgets. For the X/L/M/S variants, DEIMv2 adopts DINOv3-pretrained or DINOv3-distilled ViT backbones and introduces a **Spatial Tuning Adapter (STA)** that converts DINOv3's single-scale output into multi-scale features, complementing strong semantics with fine-grained spatial detail. The ultra-lightweight variants (`N`/`Pico`/`Femto`/`Atto`) instead use a depth- and width-pruned **HGNetv2** backbone to meet strict resource budgets. Combined with a simplified decoder and an upgraded Dense O2O scheme, DEIMv2 achieves a strong performance-cost trade-off across the board, with the `deimv2_s` model notably surpassing 50 AP on the challenging COCO benchmark at under 10M parameters.

> **License note:** DEIMv2 is released by Intellindust AI Lab under a **non-commercial research license** (see [LICENSE.md](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md)) — commercial use requires a separate license from Intellindust. Review the upstream license terms before deploying these weights in a commercial product.

---

## Model Variants

| Model | Backbone | Input Size | Params(M) | mAP[.5:.95]% | Validated Devices | Config |
|-------|----------|------------|-----------|--------------|--------------------|--------|
| `deimv2_atto` | HGNetv2-Atto | 320×320 | 0.5 | 23.8 | N/A | N/A |
| `deimv2_femto` | HGNetv2-Femto | 416×416 | 1.0 | 31.0 | N/A | N/A |
| `deimv2_pico` | HGNetv2-Pico | 640×640 | 1.5 | 38.5 | N/A | N/A |
| `deimv2_n` | HGNetv2-B0 | 640×640 | 3.6 | 43.0 | N/A | N/A |
| `deimv2_s` | DINOv3-vit_tiny | 640×640 | 9.7 | 50.9 | TDA4VH | [deimv2_s_config.yaml](deimv2_s_config.yaml) |
| `deimv2_m` | DINOv3-vit_tinyplus | 640×640 | 18.1 | 53.0 | TDA4VH | [deimv2_m_config.yaml](deimv2_m_config.yaml) |
| `deimv2_l` | DINOv3-vit_small | 640×640 | 32.2 | 56.0 | N/A | N/A |
| `deimv2_x` | DINOv3-vit_small+ | 640×640 | 50.3 | 57.8 | N/A | N/A |

> mAP values are on COCO val2017, as reported by the upstream [DEIMv2 repository](https://github.com/Intellindust-AI-Lab/DEIMv2).

**Recommended for edge deployment:** `deimv2_s` (best accuracy/compute trade-off; the only variant marked `recommended: true` in its TIDL config)

---

## Quick Start

### Prerequisites

```bash
# Install core dependencies (auto-installed by prepare_model.py if missing)
pip install torch>=1.12.0 torchvision>=0.13.0 onnx>=1.14.0 huggingface_hub timm calflops

# ONNX inference
pip install onnxruntime>=1.15.0

# scipy is required because DEIM imports it at module load time (models/matcher.py)
pip install scipy
```

> **Note:** `prepare_model.py` uses `huggingface_hub`, which requires **git** and **internet access** on first use to clone the DEIMv2 source and download pretrained weights (~10–200 MB from HuggingFace Hub). Subsequent runs reuse the cache at `~/.cache/deimv2_src` and `~/.cache/huggingface/hub`.

### Export the Model

Pretrained COCO weights are downloaded automatically via `huggingface_hub` on first use.

```bash
# List all available variants with accuracy info
python prepare_model.py --list-models

# Export the default model (deimv2_s)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model deimv2_m

# Export multiple variants at once
python prepare_model.py --model deimv2_s deimv2_m deimv2_l

# Export all supported models
python prepare_model.py --model all

# Export with a custom input resolution
python prepare_model.py --model deimv2_s --shape 800 800

# Export from a locally trained checkpoint
python prepare_model.py --model deimv2_s --weights /path/to/checkpoint.pth

# Force re-export even if the .onnx already exists
python prepare_model.py --model deimv2_s --force
```

The script automatically:
- Installs missing dependencies (`torch`, `onnx`, `huggingface_hub`, `timm`, `scipy`) if not present
- Clones the DEIMv2 source repository via git on first use (cached at `~/.cache/deimv2_src`)
- Downloads the pretrained COCO weights for the requested variant(s) from HuggingFace Hub
- Wraps the model (backbone + encoder + decoder, skipping the postprocessor) to accept a plain `(N, 3, H, W)` tensor
- Exports to ONNX (opset 17 by default) with constant folding and shape inference
- Simplifies the graph with `onnxslim`/`onnxsim` (best-effort) and saves `<model_key>.onnx` in the output directory

**ONNX model inputs / outputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `images` (input) | `(N, 3, H, W)` | ImageNet-normalized float32 |
| `pred_boxes` (output 0) | `(N, num_queries, 4)` | Boxes as (cx, cy, w, h), normalized [0, 1] |
| `pred_logits` (output 1) | `(N, num_queries, 80)` | Raw class logits for 80 COCO classes |

DEIMv2 outputs a fixed number of query slots per image (100–300 depending on variant) regardless of the number of objects present.

**Input preprocessing** — DEIMv2 expects ImageNet-normalized inputs:

```python
import cv2
import numpy as np

mean  = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
scale = np.array([0.017125, 0.017507, 0.017429], dtype=np.float32)  # 1/255 / std

img = cv2.imread("image.jpg")               # BGR uint8
h, w = MODEL_SHAPE                          # e.g. (640, 640) for S/M/L/X; (320,320) atto; (416,416) femto
img = cv2.resize(img, (w, h))
img = img.astype(np.float32)
img = (img - mean) * scale
img = np.transpose(img, (2, 0, 1))          # HWC → CHW
img = np.expand_dims(img, 0)                # add batch dim → (1, 3, H, W)
```

**Post-processing** — class scores are computed via **sigmoid** (not softmax):

```python
import numpy as np

CONFIDENCE_THRESHOLD = 0.25

def postprocess(pred_boxes, pred_logits, image_h, image_w, threshold=CONFIDENCE_THRESHOLD):
    boxes  = pred_boxes[0]                      # (num_queries, 4) cx,cy,w,h normalized
    logits = pred_logits[0]                     # (num_queries, 80)

    scores = 1 / (1 + np.exp(-logits))          # sigmoid
    scores = scores.max(axis=1)
    labels = scores.argmax(axis=1)
    keep   = scores > threshold

    cx, cy, bw, bh = boxes[keep].T
    x1 = (cx - bw / 2) * image_w
    y1 = (cy - bh / 2) * image_h
    x2 = (cx + bw / 2) * image_w
    y2 = (cy + bh / 2) * image_h

    return np.stack([x1, y1, x2, y2], axis=1), labels[keep], scores[keep]
```

### Compile and Infer uing TIDL Runner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using TIDL Runner - on PC**

```bash
cd /path/to/tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/deimv2_s_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/deimv2_s_config.yaml
```

> Replace `deimv2_s_config.yaml` with `deimv2_m_config.yaml` to compile/infer the `deimv2_m` variant. To evaluate accuracy instead of just compiling, replace `compile` with `evaluate`.

### Compile and Infer using TIDL Tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

---

## Citation

If you use these models, please cite:

```bibtex
@article{huang2025deimv2,
  title   = {Real-Time Object Detection Meets DINOv3},
  author  = {Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal = {arXiv preprint arXiv:2509.20787},
  year    = {2025}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2509.20787](https://arxiv.org/abs/2509.20787) |
| **Source Code** | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) |
| **License** | [LICENSE.md (non-commercial)](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md) |
| **HGNetv2 Backbone** | [Peterande/HGNetv2](https://github.com/Peterande/HGNetv2) |
| **DINOv3 Backbone** | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) |
| **COCO Dataset** | [cocodataset.org](https://cocodataset.org) |
| **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |

---

## Related Models

<table>
<tr>
<td align="center">

**DETR**
Original end-to-end
DETR transformer detector

</td>
<td align="center">

**Deformable-DETR**
Deformable attention
for faster convergence

</td>
<td align="center">

**RT-DETRv2**
Real-time DETR
transformer detector

</td>
<td align="center">

**RF-DETR**
Real-time DETR
with flexible backbones

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
