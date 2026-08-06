---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

# Deformable DETR for TI EdgeAI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange.svg)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection-green.svg)](https://github.com/TexasInstruments/edgeai)

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Citation](#citation)
- [Additional Resources](#additional-resources)

## Introduction

**Ready-to-deploy object detection for TI edge devices**

This Deformable DETR model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge.

Deformable DETR (Deformable Transformers for End-to-End Object Detection) is a transformer-based detector from SenseTime / fundamentalvision that addresses the slow convergence and limited feature resolution of the original DETR. Its key innovation is a **deformable attention module** that attends to only a small set of key sampling points around a reference point rather than all feature map positions, reducing complexity from O(H²W²) to O(HW). This allows the use of multi-scale feature maps and achieves better AP — especially on small objects — with **10× fewer training epochs** than DETR. Five variants are provided, ranging from a lightweight single-scale model to a two-stage design with iterative bounding box refinement.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

### Detection Models

| Dataset | Model Name                                      | Model ID     | Input Size | mAP[.5:.95]% | Params (M) | FLOPs (G) | Notes                     |
| -       | -                                               | -            | -          | -            | -          | -         | -                         |
| COCO    | deformable-detr-r50-ss                          | `od-mh8060`  | 800×800    | 39.4         | 34         | 78        | Single-scale              |
| COCO    | deformable-detr-r50-ss-dc5                      | `od-mh8061`  | 800×800    | 41.5         | 34         | 128       | Single-scale + DC5 ⚠️      |
| COCO    | deformable-detr-r50                             | `od-mh8062`  | 800×800    | 44.5         | 40         | 173       | Multi-scale, **Recommended** |
| COCO    | deformable-detr-r50-plus-iterbox                | `od-mh8063`  | 800×800    | 46.2         | 41         | 173       | + iterative bbox refine   |
| COCO    | deformable-detr-r50-two-stage                   | `od-mh8064`  | 800×800    | 46.9         | 41         | 173       | Two-stage + bbox refine   |

> mAP values on COCO val2017. Inference speed measured on NVIDIA V100 GPU.  
> ⚠️ **DC5 variant (`od-mh8061`) is currently disabled** — TIDL does not support dilated convolutions in ResNet. The config file is provided for reference.  
> All variants use a ResNet-50 backbone pretrained on ImageNet.

### ONNX Model Inputs / Outputs

| Tensor            | Shape              | Description                                         |
| -                 | -                  | -                                                   |
| `images` (input)  | `(N, 3, H, W)`     | ImageNet-normalized float32, typically 800×800      |
| `pred_boxes` (output 0) | `(N, 300, 4)` | Boxes as (cx, cy, w, h), normalized [0, 1]     |
| `pred_logits` (output 1)| `(N, 300, 91)`| Class logits for 91 COCO classes (sigmoid-based) |

Deformable DETR uses **300 query slots** (vs. 100 in DETR) and **focal loss with sigmoid** classification (no explicit background class). Post-processing applies a sigmoid threshold rather than softmax + background filtering.

---

## Setup & Installation

### Requirements

```bash
# Core dependencies (auto-installed by prepare_model.py if missing)
pip install torch>=1.12.0 torchvision>=0.13.0 onnx>=1.14.0 scipy gdown>=5.2.0

# ONNX inference
pip install onnxruntime>=1.15.0
```

> **Note:** `prepare_model.py` clones the [Deformable-DETR repository](https://github.com/fundamentalvision/Deformable-DETR) into `~/.cache/deformable_detr` on first use and downloads pretrained weights (~150–200 MB) from **Google Drive** via `gdown`. No CUDA compilation is required — the script installs a pure-Python fallback for the deformable attention module.

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

Pretrained COCO weights are downloaded automatically from Google Drive on first use. Use `prepare_model.py` to export any variant to ONNX.

```bash
# List all available variants with accuracy and parameter info
python prepare_model.py --list-models

# Export the default model (deformable_detr – multi-scale, recommended)
python prepare_model.py

# Export specific variants
python prepare_model.py --model deformable_detr_single_scale
python prepare_model.py --model deformable_detr
python prepare_model.py --model deformable_detr_plus_iterative_bbox_refinement
python prepare_model.py --model deformable_detr_two_stage

# Export multiple variants at once
python prepare_model.py --model deformable_detr deformable_detr_two_stage

# Export all variants (skips any already exported)
python prepare_model.py --model all

# Use HuggingFace Hub instead of Google Drive (recommended on corporate networks)
python prepare_model.py --method optimum
python prepare_model.py --method optimum --model all

# Export with a custom input resolution
python prepare_model.py --model deformable_detr --shape 640 640

# Export from a locally downloaded checkpoint
python prepare_model.py --model deformable_detr --weights /path/to/checkpoint.pth

# Export with a specific ONNX opset version
python prepare_model.py --model deformable_detr --opset 18

# Save to a custom output directory
python prepare_model.py --model deformable_detr --output-dir ./exports

# Force re-export and re-download even if files already exist
python prepare_model.py --model deformable_detr --force

# Force re-clone of the source repository
python prepare_model.py --model deformable_detr --force-reclone

# Skip onnx-simplifier step
python prepare_model.py --model deformable_detr --skip-simplify
```

The script will automatically:
1. Install missing dependencies (torch, torchvision, onnx, scipy, gdown) if not present
2. Clone the Deformable-DETR repository from GitHub into `~/.cache/deformable_detr` on first use (or download from HuggingFace Hub with `--method optimum`)
3. Install a pure-Python fallback for the multi-scale deformable attention module (no CUDA compilation required)
4. Download pretrained COCO weights from Google Drive via `gdown` (or from HuggingFace Hub with `--method optimum`)
5. Build the model from source with the correct architecture flags
6. Wrap the model to accept a plain `(N, 3, H, W)` tensor (handles `NestedTensor` internally)
7. Export to ONNX with constant folding enabled
8. Simplify the ONNX graph with onnx-simplifier (unless `--skip-simplify`)
9. Fix float64 nodes for TIDL compatibility (removes Cast-to-DOUBLE, updates stale type annotations)
10. Validate the exported ONNX graph
11. Save as `<model_key>.onnx` in the output directory

> **Corporate Network Tip:** If Google Drive is blocked by your proxy, use `--method optimum` to download weights directly from HuggingFace Hub instead.

#### Export script parameters

| Flag               | Default                  | Description                                               |
| -                  | -                        | -                                                         |
| `--model`          | `deformable_detr`        | One or more variant names, or `all` to export every missing variant. See `--list-models`. |
| `--method`         | `torch`                  | Export method: `torch` (Google Drive weights) or `optimum` (HuggingFace Hub, proxy-friendly). |
| `--shape H W`      | `800 800`                | Custom input resolution (height width).                   |
| `--opset`          | `17`                     | ONNX opset version.                                       |
| `--batch-size`     | `1`                      | Batch size embedded in the exported graph.                |
| `--weights`        | COCO pretrained          | Path to a local `.pth` checkpoint (ignored with `--method optimum`). |
| `--output-dir`     | Script directory         | Directory where `.onnx` and `.pth` files are saved.       |
| `--force`          | `False`                  | Re-export and re-download even if files already exist.    |
| `--force-reclone`  | `False`                  | Re-clone the source repository (removes cached copy).     |
| `--skip-simplify`  | `False`                  | Skip the onnx-simplifier step.                            |
| `--quiet`          | `False`                  | Suppress verbose progress messages.                       |
| `--list-models`    | —                        | Print model catalogue table and exit.                     |

### Input Preprocessing

Deformable DETR expects ImageNet-normalized inputs:

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

### Post-processing

Deformable DETR uses **sigmoid + focal loss** (not softmax). There is no explicit background class; low-confidence predictions are filtered by a score threshold.

```python
import numpy as np

CONFIDENCE_THRESHOLD = 0.5

def postprocess(pred_boxes, pred_logits, image_h, image_w, threshold=CONFIDENCE_THRESHOLD):
    # pred_boxes  : (1, 300, 4) – cx,cy,w,h normalized [0,1]
    # pred_logits : (1, 300, 91) – per-class logits (apply sigmoid)

    boxes  = pred_boxes[0]                     # (300, 4)
    logits = pred_logits[0]                    # (300, 91)

    # Apply sigmoid to get per-class probabilities
    probs  = 1.0 / (1.0 + np.exp(-logits))    # (300, 91)
    scores = probs.max(-1)                     # max score per query
    labels = probs.argmax(-1)                  # class with highest score

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

# deformable_detr_single_scale (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path deformable_detr_single_scale_config.yaml

# deformable_detr (multi-scale, recommended, 800×800)
tidlrunner-cli compile --target_device J784S4 --config_path deformable_detr_config.yaml

# deformable_detr_plus_iterative_bbox_refinement (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path deformable_detr_plus_iterative_bbox_refinement_config.yaml

# deformable_detr_two_stage (800×800)
tidlrunner-cli compile --target_device J784S4 --config_path deformable_detr_two_stage_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

---

## Citation

```bibtex
@article{zhu2020deformable,
  title   = {Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author  = {Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and
             Wang, Xiaogang and Dai, Jifeng},
  journal = {arXiv preprint arXiv:2010.04159},
  year    = {2020}
}
```

---

## Additional Resources

- [Deformable DETR GitHub Repository](https://github.com/fundamentalvision/Deformable-DETR) (Apache 2.0)
- [Deformable DETR Paper (arXiv)](https://arxiv.org/abs/2010.04159)
- [COCO Dataset](https://cocodataset.org)
- [TI EdgeAI Model Zoo](https://github.com/TexasInstruments/edgeai)
- [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner)
- [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

**License:** Apache 2.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026 (v2: added `--model all`, `--method optimum`, float64 TIDL fix)
