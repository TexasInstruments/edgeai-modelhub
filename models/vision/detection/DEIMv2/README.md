# DEIMv2 for TI EdgeAI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange.svg)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green.svg)](https://github.com/TexasInstruments/edgeai)

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Input Preprocessing](#input-preprocessing)
- [Post-processing](#post-processing)
- [Using the model on TI device](#using-the-model-on-ti-device)
- [Citation](#citation)
- [Additional Resources](#additional-resources)

## Introduction

**Ready-to-deploy real-time object detection for TI edge devices**

This DEIMv2 model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready detection with minimal setup.

DEIMv2 (DEtection with Input Mixup v2) is an evolution of the DEIM framework leveraging rich features from DINOv3. Our method provides various model sizes from ultra-light (Atto) to extra-large (X) to be adaptable for a wide range of scenarios. Across these variants, DEIMv2 achieves state-of-the-art performance, with the S-sized model notably surpassing 50 AP on the challenging COCO benchmark.

The repository is archived (Apache 2.0), with eight detection variants available as pretrained COCO models.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

### Detection Models

| Dataset | Model Name             | Model ID    | Input Size | mAP[.5:.95]% | mAP[.50]% | FLOPs(G) | Params(M) | Latency (ms) | Backbone                |
|---------|------------------------|-------------|------------|--------------|-----------|----------|-----------|--------------|-------------------------|
| COCO    | deimv2_atto            | `od-mhTBD`  | 320×320    | 23.8         | 35.7      | 0.8      | 0.5       | 1.10         | HGNetv2-Atto            |
| COCO    | deimv2_femto           | `od-mhTBD`  | 416×416    | 31.0         | 45.6      | 1.7      | 1.0       | 1.45         | HGNetv2-Femto           |
| COCO    | deimv2_pico            | `od-mhTBD`  | 640×640    | 38.5         | 54.8      | 5.2      | 1.5       | 2.13         | HGNetv2-Pico            |
| COCO    | deimv2_n               | `od-mhTBD`  | 640×640    | 43.0         | 60.3      | 6.8      | 3.6       | 2.32         | HGNetv2-B0              |
| COCO    | deimv2_s               | `od-mhTBD`  | 640×640    | 50.9         | 68.6      | 25.6     | 9.7       | 5.78         | DINOv3-vit_tiny         |
| COCO    | deimv2_m               | `od-mhTBD`  | 640×640    | 53.0         | 70.5      | 52.2     | 18.1      | 8.80         | DINOv3-vit_tinyplus     |
| COCO    | deimv2_l               | `od-mhTBD`  | 640×640    | 56.0         | 73.2      | 96.7     | 32.2      | 10.47        | DINOv3-vit_small        |
| COCO    | deimv2_x               | `od-mhTBD`  | 640×640    | 57.8         | 74.6      | 151.6    | 50.3      | 13.75        | DINOv3-vit_small+       |

> mAP values are on COCO val2017. Latency measured on NVIDIA A100 GPU with TensorRT FP16.

### ONNX Model Inputs / Outputs

| Tensor              | Shape                   | Description |
|---------------------|-------------------------|-------------|
| `images` (input)    | `(N, 3, H, W)`          | ImageNet-normalized float32 |
| `pred_boxes` (output 0) | `(N, 300, 4)`       | Boxes as (cx, cy, w, h), normalized [0, 1] |
| `pred_logits` (output 1) | `(N, 300, 80)`      | Raw class logits for 80 COCO classes |

DEIMv2 outputs exactly **300 query slots** per image regardless of the number of objects. Post-processing applies sigmoid to logits and filters low-confidence predictions.

---

## Setup & Installation

### Requirements

```bash
# Install core dependencies (auto-installed by prepare_model.py if missing)
pip install torch>=1.12.0 torchvision>=0.13.0 onnx>=1.14.0 huggingface_hub timm calflops

# ONNX inference
pip install onnxruntime>=1.15.0

# scipy is required because DEIM imports it at module load time (models/matcher.py)
pip install scipy
```

> **Note:** `prepare_model.py` uses `huggingface_hub` which requires **git** and **internet access** on first use to download pretrained weights (~10–200 MB from HuggingFace Hub). Subsequent runs use the cached copy in `~/.cache/huggingface/hub`.

### For TI hardware deployment

Refer to **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)** for setup.

---

## Usage

### Download from HuggingFace

If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Export

Pretrained COCO weights are downloaded automatically via `huggingface_hub` on first use. Use `prepare_model.py` to export any variant to ONNX.

```bash
# List all available variants with accuracy info
python prepare_model.py --list-models

# Export the default model (deimv2_s)
python prepare_model.py

# Export specific variants
python prepare_model.py --model deimv2_atto
python prepare_model.py --model deimv2_femto
python prepare_model.py --model deimv2_pico
python prepare_model.py --model deimv2_n
python prepare_model.py --model deimv2_s
python prepare_model.py --model deimv2_m
python prepare_model.py --model deimv2_l
python prepare_model.py --model deimv2_x

# Export multiple variants at once
python prepare_model.py --model deimv2_s deimv2_m deimv2_l

# Export with a custom input resolution
python prepare_model.py --model deimv2_s --shape 800 800

# Export from a locally trained checkpoint
python prepare_model.py --model deimv2_s --weights /path/to/checkpoint.pth

# Export with a specific ONNX opset version
python prepare_model.py --model deimv2_s --opset 18

# Save to a custom output directory
python prepare_model.py --model deimv2_s --output-dir ./exports

# Force re-export even if .onnx already exists
python prepare_model.py --model deimv2_s --force

# Apply onnx-simplifier after export (best-effort)
python prepare_model.py --model deimv2_s --simplify
```

The script will automatically:
1. Install missing dependencies (torch, onnx, huggingface_hub, timm, scipy) if not present
2. Clone the DEIMv2 repository via git on first use
3. Download pretrained COCO weights from HuggingFace Hub on first use
4. Wrap the model to accept a plain `(N, 3, H, W)` tensor
5. Export to ONNX with constant folding enabled
6. Validate the exported ONNX graph (if `onnx` package is available)
7. Save as `<model_key>.onnx` in the output directory

### Input Preprocessing

DEIMv2 expects ImageNet-normalized inputs (same as used in training):

```python
import cv2
import numpy as np

# ImageNet mean/std in BGR format (OpenCV loads as BGR)
mean  = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
scale = np.array([0.017125, 0.017507, 0.017429], dtype=np.float32)  # 1/255 / std

img = cv2.imread("image.jpg")               # BGR uint8
# Resize to model's expected input size (varies by model)
h, w = MODEL_SHAPE  # e.g., (640, 640) for S/M/L/X variants
img = cv2.resize(img, (w, h))
img = img.astype(np.float32)               # uint8 → float32
img = (img - mean) * scale                 # normalize
img = np.transpose(img, (2, 0, 1))         # HWC → CHW
img = np.expand_dims(img, 0)               # add batch dim → (1, 3, H, W)
```

> **Note:** Models ending in `_atto` (320×320) and `_femto` (416×416) use smaller input sizes.

### Post-processing

DEIMv2 outputs class probabilities via **sigmoid** (not softmax). A typical post-processing pipeline:

```python
import numpy as np

CONFIDENCE_THRESHOLD = 0.25  # Adjust as needed

def postprocess(pred_boxes, pred_logits, image_h, image_w, threshold=CONFIDENCE_THRESHOLD):
    # pred_boxes  : (1, 300, 4) – cx,cy,w,h normalized [0,1]
    # pred_logits : (1, 300, 80) – class logits

    boxes   = pred_boxes[0]    # (300, 4)
    logits  = pred_logits[0]   # (300, 80)

    # Apply sigmoid and get max score/class per query
    scores = 1 / (1 + np.exp(-logits))          # sigmoid
    scores = scores.max(axis=1)                 # max score per query
    labels = scores.argmax(axis=1)              # class with max prob

    # Filter low-confidence predictions
    keep = scores > threshold

    # Convert cx,cy,w,h → x1,y1,x2,y2 in pixel coordinates
    cx, cy, bw, bh = boxes[keep].T
    x1 = (cx - bw / 2) * image_w
    y1 = (cy - bh / 2) * image_h
    x2 = (cx + bw / 2) * image_w
    y2 = (cy + bh / 2) * image_h

    return np.stack([x1, y1, x2, y2], axis=1), labels[keep], scores[keep]
```

### Using the model on TI device

#### Option 1: Advanced Users (TIDL Tools)
For users familiar with TIDL and requiring fine-grained control:

```bash
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tools
```

Refer to the [tidl-tools setup](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/README.md#getting-started) page for more details on compile and infer.

#### Option 2: Simplified Workflow (Recommended)
Setup tidl runner using this link: [edgeai-tidlrunner setup](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tidlrunner/docs/setup.md)

```bash
# Compile model and evaluate performance on J784S4.

# deimv2_atto (320×320)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_atto_config.yaml

# deimv2_femto (416×416)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_femto_config.yaml

# deimv2_pico (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_pico_config.yaml

# deimv2_n (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_n_config.yaml

# deimv2_s (640×640) [default]
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_s_config.yaml

# deimv2_m (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_m_config.yaml

# deimv2_l (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_l_config.yaml

# deimv2_x (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path deimv2_x_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

---

## Citation

```bibtex
@inproceedings{huang2026deimv2,
  title     = {DEIMv2: Real-Time Object Detection Meets DINOv3},
  author    = {Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Additional Resources

- [DEIMv2 GitHub Repository](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [DEIMv2 Paper (arXiv)](https://arxiv.org/abs/xxxx.xxxxx)
- [HGNetv2 Backbone](https://github.com/Peterande/HGNetv2)
- [DINOv3 ViT Backbone](https://github.com/facebookresearch/dinov3)
- [COCO Dataset](https://cocodataset.org)
- [TI EdgeAI Model Zoo](https://github.com/TexasInstruments/edgeai)
- [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner)
- [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)
