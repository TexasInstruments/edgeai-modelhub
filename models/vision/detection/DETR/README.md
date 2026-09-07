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

# DETR for TI EdgeAI

### Set-Prediction Object Detection and Panoptic Segmentation via Transformers

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Detection%20%7C%20Segmentation-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-COCO-blueviolet?style=for-the-badge)](https://cocodataset.org)

</div>

---

## Overview

**DETR** (DEtection TRansformer) is a transformer-based object detection architecture from Facebook Research that eliminates the need for hand-crafted components like anchor generation and NMS post-processing. It reformulates object detection as a direct set-prediction problem, using bipartite matching together with a transformer encoder-decoder to predict a fixed set of 100 object queries per image.

DETR matches Faster R-CNN with a ResNet-50 backbone in AP while using half the FLOPs, and extends naturally to panoptic segmentation by adding a mask head on top of the detection queries. The original [facebookresearch/detr](https://github.com/facebookresearch/detr) repository is archived (Apache 2.0), and pretrained COCO weights are pulled automatically via `torch.hub`.

This export covers the four ResNet-backbone **detection** variants (`detr_resnet50`, `detr_resnet50_dc5`, `detr_resnet101`, `detr_resnet101_dc5`). The panoptic segmentation variants (`detr_resnet50_panoptic`, `detr_resnet50_dc5_panoptic`, `detr_resnet101_panoptic`) are supported by the upstream repository and by `prepare_model.py`, but are not included as pre-exported artifacts in this folder.

---

## Model Variants

| Model | Backbone | mAP[.5:.95]% | mAP[.50]% | Validated Devices | Config |
|-------|----------|-------------|-----------|--------------------|--------|
| `detr_resnet50` | ResNet-50 | 42.0 | 62.4 | TDA4VH | [detr_resnet50_config.yaml](detr_resnet50_config.yaml) |
| `detr_resnet50_dc5` | ResNet-50 DC5 | 43.3 | 63.1 | TDA4VH | [detr_resnet50_dc5_config.yaml](detr_resnet50_dc5_config.yaml) |
| `detr_resnet101` | ResNet-101 | 43.5 | 63.8 | TDA4VH | [detr_resnet101_config.yaml](detr_resnet101_config.yaml) |
| `detr_resnet101_dc5` | ResNet-101 DC5 | 44.9 | 64.7 | TDA4VH | [detr_resnet101_dc5_config.yaml](detr_resnet101_dc5_config.yaml) |

> mAP values are on COCO val2017. DC5 = dilated convolutions in the last ResNet block (stride 16→32 kept at stride 8→16), giving higher-resolution features at the cost of higher compute.

**Recommended for edge deployment:** `detr_resnet50` (best accuracy/compute trade-off)

---

## Quick Start

### Prerequisites

```bash
pip install torch>=1.12.0 torchvision>=0.13.0 onnx>=1.14.0 scipy
pip install onnxruntime>=1.15.0
```

`scipy` is required because DETR imports it at module load time (`models/matcher.py`). All of the above are auto-installed by `prepare_model.py` if missing.

### Export the Model

```bash
# Export the default model (detr_resnet50)
python prepare_model.py

# Export a specific model variant
python prepare_model.py --model detr_resnet101

# Export multiple variants at once
python prepare_model.py --model detr_resnet50 detr_resnet101

# List all available variants with accuracy info
python prepare_model.py --list-models

# Export from a locally trained checkpoint
python prepare_model.py --model detr_resnet50 --weights /path/to/checkpoint.pth
```

The script automatically:
- Installs missing dependencies (`torch`, `torchvision`, `onnx`, `scipy`) if not present
- Loads the pretrained model via `torch.hub` (`facebookresearch/detr:main`), cloning the DETR source and downloading pretrained COCO weights from `dl.fbaipublicfiles.com` on first use
- Wraps the model to accept a plain `(N, 3, H, W)` tensor instead of a `NestedTensor`
- Exports to ONNX (opset 17 by default) with constant folding enabled, and validates the exported graph
- Saves the result as `<model_key>.onnx` in the output directory

> **Note:** DC5 (`_dc5`) variants are currently skipped by `prepare_model.py` with a warning, since TIDL does not yet support the dilated-conv backbone for compilation. The pre-exported `.onnx`/config artifacts for these variants remain in this folder for reference.

### Compile and Infer uing edgeai-tidlrunner

> **Note:** Run the commands below from inside the `tidlrunner` directory (the cloned [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) repository), with `--config_path` pointing to this model's config file.

**Compile using edgeai-tidlrunner - on PC**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device J784S4 \
  --config_path /path/to/detr_resnet50_config.yaml
```

**Run Inference Benchmark - on device**

```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device J784S4 \
  --config_path /path/to/detr_resnet50_config.yaml
```

Swap `detr_resnet50_config.yaml` for `detr_resnet50_dc5_config.yaml`, `detr_resnet101_config.yaml`, or `detr_resnet101_dc5_config.yaml` to compile/infer the other variants.

### Compile and Infer using edgeai-tidl-tools (Advanced):

Follow the instructions at https://github.com/TexasInstruments/edgeai-tidl-tools

### Deploy using edgeai-tidl-tools:

Deplyment can be done using **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)**. For ONNX models, onnxruntime-tidl with TIDL acceleration can be used. Consult the documentation of edgeai-tidl-tools for more details.

---

## Citation

If you use these models, please cite:

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

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2005.12872](https://arxiv.org/abs/2005.12872) |
| **Source Code** | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| **Blog Post** | [End-to-End Object Detection with Transformers](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers) |
| **COCO Dataset** | [cocodataset.org](https://cocodataset.org) |
| **edgeai-tidl-tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| **edgeai-tidlrunner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| **TI EdgeAI Ecosystem** | [GitHub](https://github.com/TexasInstruments/edgeai) |

---

## Related Models

<table>
<tr>
<td align="center">

**Deformable-DETR**
Deformable attention
Faster convergence

</td>
<td align="center">

**RT-DETRv2**
Real-time transformer
NMS-free detection

</td>
<td align="center">

**RF-DETR**
Receptive-field DETR
Lightweight edge variant

</td>
<td align="center">

**DEIMv2**
Improved DETR training
Higher accuracy/epoch

</td>
</tr>
</table>

---

<div align="center">

**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

</div>
