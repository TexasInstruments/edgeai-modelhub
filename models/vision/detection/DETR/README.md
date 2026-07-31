---
license: apache-2.0
tags:
- vision
- image-detection
- image-segmentation
datasets:
- COCO
---

# DETR for TI EdgeAI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange.svg)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection%20%7C%20Panoptic%20Segmentation-green.svg)](https://github.com/TexasInstruments/edgeai)

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Citation](#citation)
- [Additional Resources](#additional-resources)

## Introduction

**Ready-to-deploy object detection and panoptic segmentation for TI edge devices**

This DETR model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready detection and segmentation with minimal setup.

DETR (DEtection TRansformer) is a transformer-based object detection architecture from Facebook Research that eliminates the need for hand-crafted components like anchor generation and NMS post-processing. It approaches object detection as a direct set prediction problem using bipartite matching and a transformer encoder-decoder. DETR matches Faster R-CNN with ResNet-50 in AP while using half the FLOPs, and extends naturally to panoptic segmentation.

The repository is archived (Apache 2.0), with four detection and three panoptic segmentation variants available as pretrained COCO models.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

### Detection Models

| Dataset | Model Name             | Model ID    | Input Size | mAP[.5:.95]% | mAP[.50]% | Backbone     | Notes              |
| -       | -                      | -           | -          | -            | -         | -            | -                  |
| COCO    | detr-resnet50          | `od-mhTBD`  | 800×800    | 42.0         | 62.4      | ResNet-50    | Recommended        |
| COCO    | detr-resnet50-dc5      | `od-mhTBD`  | 800×800    | 43.3         | 63.1      | ResNet-50    | Dilated conv       |
| COCO    | detr-resnet101         | `od-mhTBD`  | 800×800    | 43.5         | 63.8      | ResNet-101   |                    |
| COCO    | detr-resnet101-dc5     | `od-mhTBD`  | 800×800    | 44.9         | 64.7      | ResNet-101   | Dilated conv       |

### Panoptic Segmentation Models

| Dataset | Model Name                     | Model ID    | Input Size | Box AP[.5:.95]% | PQ    | Backbone   | Notes              |
| -       | -                              | -           | -          | -               | -     | -          | -                  |
| COCO    | detr-resnet50-panoptic         | `od-mhTBD`  | 800×800    | 38.8            | 43.4  | ResNet-50  | Recommended        |
| COCO    | detr-resnet50-dc5-panoptic     | `od-mhTBD`  | 800×800    | 40.2            | 44.6  | ResNet-50  | Dilated conv       |
| COCO    | detr-resnet101-panoptic        | `od-mhTBD`  | 800×800    | 40.1            | 45.1  | ResNet-101 |                    |

> mAP and PQ values are on COCO val2017. Latency measured on V100 GPU with TorchScript.  
> DC5 = dilated convolutions in the last ResNet block (stride 16→32 kept at stride 8→16), giving higher-resolution features at the cost of higher compute.

### ONNX Model Inputs / Outputs

| Tensor | Shape | Description |
| - | - | - |
| `images` (input) | `(N, 3, H, W)` | ImageNet-normalized float32 |
| `pred_boxes` (output 0) | `(N, 100, 4)` | Boxes as (cx, cy, w, h), normalized [0, 1] |
| `pred_logits` (output 1) | `(N, 100, 92)` | Class logits for 91 COCO classes + no-object |
| `pred_masks` (output 2, panoptic only) | `(N, 100, H/4, W/4)` | Per-query panoptic mask logits |

DETR outputs exactly **100 query slots** per image regardless of the number of objects. Post-processing filters out slots where the `no-object` class (index 91 for detection, 250 for panoptic) has the highest softmax probability.

---

## Setup & Installation

### Requirements

```bash
# Install core dependencies (auto-installed by prepare_model.py if missing)
pip install torch>=1.12.0 torchvision>=0.13.0 onnx>=1.14.0 scipy

# ONNX inference
pip install onnxruntime>=1.15.0

# scipy is required because DETR imports it at module load time (models/matcher.py)
```

> **Note:** `prepare_model.py` uses `torch.hub.load` which requires **git** and **internet access** on first use to clone the DETR repository (~5 MB) and download pretrained weights (~160–240 MB from `dl.fbaipublicfiles.com`). Subsequent runs use the cached copy in `~/.cache/torch/hub/`.

#### For TI hardware deployment
Refer to **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)** for setup.

---

## Usage

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Export

Pretrained COCO weights are downloaded automatically via torch.hub on first use. Use `prepare_model.py` to export any variant to ONNX.

```bash
# List all available variants with accuracy info
python prepare_model.py --list-models

# Export the default model (detr_resnet50)
python prepare_model.py

# Export specific detection variants
python prepare_model.py --model detr_resnet50
python prepare_model.py --model detr_resnet50_dc5
python prepare_model.py --model detr_resnet101
python prepare_model.py --model detr_resnet101_dc5

# Export multiple variants at once
python prepare_model.py --model detr_resnet50 detr_resnet101

# Export panoptic segmentation variants
python prepare_model.py --model detr_resnet50_panoptic
python prepare_model.py --model detr_resnet50_panoptic detr_resnet101_panoptic

# Export with a custom input resolution
python prepare_model.py --model detr_resnet50 --shape 800 1333

# Export from a locally trained checkpoint
python prepare_model.py --model detr_resnet50 --weights /path/to/checkpoint.pth

# Export with a specific ONNX opset version
python prepare_model.py --model detr_resnet50 --opset 18

# Save to a custom output directory
python prepare_model.py --model detr_resnet50 --output-dir ./exports

# Force re-export even if .onnx already exists
python prepare_model.py --model detr_resnet50 --force

# Force re-download of DETR source and weights from hub
python prepare_model.py --model detr_resnet50 --force-hub-reload
```

The script will automatically:
1. Install missing dependencies (torch, torchvision, onnx, scipy) if not present
2. Clone the DETR repository via `torch.hub` on first use
3. Download pretrained COCO weights from `dl.fbaipublicfiles.com` on first use
4. Wrap the model to accept a plain `(N, 3, H, W)` tensor
5. Export to ONNX with constant folding enabled
6. Validate the exported ONNX graph (if `onnx` package is available)
7. Save as `<model_key>.onnx` in the output directory

#### Export script parameters

| Flag                | Default              | Description                                                    |
| -                   | -                    | -                                                              |
| `--model`           | `detr_resnet50`      | One or more variant names. See `--list-models`.                |
| `--shape H W`       | `800 800`            | Custom input resolution (height width).                        |
| `--opset`           | `17`                 | ONNX opset version.                                            |
| `--batch-size`      | `1`                  | Batch size embedded in the exported graph.                     |
| `--weights`         | COCO pretrained      | Path to a local `.pth` checkpoint.                             |
| `--output-dir`      | Script directory     | Directory where `.onnx` files are saved.                       |
| `--force`           | `False`              | Re-export even if the `.onnx` file already exists.             |
| `--force-hub-reload`| `False`              | Re-download DETR repo and weights from torch.hub.              |
| `--quiet`           | `False`              | Suppress torch.hub download messages.                          |
| `--list-models`     | —                    | Print model catalogue table and exit.                          |

### Input Preprocessing

DETR expects ImageNet-normalized inputs:

```python
import cv2
import numpy as np

mean  = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
scale = np.array([0.017125, 0.017507, 0.017429], dtype=np.float32)

img = cv2.imread("image.jpg")               # BGR uint8
img = cv2.resize(img, (800, 800))
img = img[:, :, ::-1].astype(np.float32)   # BGR → RGB
img = (img - mean) * scale                  # normalize
img = np.transpose(img, (2, 0, 1))          # HWC → CHW
img = np.expand_dims(img, 0)                # add batch dim → (1, 3, 800, 800)
```

> **Note:** For best accuracy, consider resizing such that the shorter side is 800 px and padding the longer side (as done during COCO training) rather than squeezing to 800×800.

### Post-processing

DETR outputs class probabilities via **softmax** (not sigmoid). The no-object class is the last index (91 for detection, 250 for panoptic). A typical post-processing pipeline:

```python
import numpy as np

CONFIDENCE_THRESHOLD = 0.7
NO_OBJECT_INDEX      = 91  # last class in 92-class detection output

def postprocess(pred_boxes, pred_logits, image_h, image_w, threshold=CONFIDENCE_THRESHOLD):
    # pred_boxes  : (1, 100, 4) – cx,cy,w,h normalized [0,1]
    # pred_logits : (1, 100, 92) – class logits

    boxes   = pred_boxes[0]    # (100, 4)
    logits  = pred_logits[0]   # (100, 92)

    # Apply softmax and drop the no-object class
    exp_logits = np.exp(logits - logits.max(-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(-1, keepdims=True)  # (100, 92)
    scores = probs[:, :NO_OBJECT_INDEX].max(-1)             # max score per query
    labels = probs[:, :NO_OBJECT_INDEX].argmax(-1)          # class with max prob

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

# detr_resnet50 (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path detr_resnet50_config.yaml

# detr_resnet50_dc5 (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path detr_resnet50_dc5_config.yaml

# detr_resnet101 (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path detr_resnet101_config.yaml

# detr_resnet101_dc5 (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path detr_resnet101_dc5_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

---

## Citation

```bibtex
@inproceedings{carion2020end,
  title     = {End-to-End Object Detection with Transformers},
  author    = {Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and
               Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

---

## Additional Resources

- [DETR GitHub Repository](https://github.com/facebookresearch/detr) (archived, Apache 2.0)
- [DETR Paper (arXiv)](https://arxiv.org/abs/2005.12872)
- [DETR Blog Post](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers)
- [COCO Dataset](https://cocodataset.org)
- [TI EdgeAI Model Zoo](https://github.com/TexasInstruments/edgeai)
- [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner)
- [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)
