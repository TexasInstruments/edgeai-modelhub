---
license: apache-2.0
tags:
- vision
- image-detection
- image-segmentation
datasets:
- COCO
---

# RF-DETR for TI EdgeAI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange.svg)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Object%20Detection%20%7C%20Segmentation-green.svg)](https://github.com/TexasInstruments/edgeai)

## Table of Contents
- [Introduction](#introduction)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Citation](#citation)
- [Additional Resources](#additional-resources)

## Introduction

🚀 **Ready-to-deploy object detection and instance segmentation for TI edge devices**

This RF-DETR model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge. Whether you're building industrial automation systems, smart cameras, robotics, or IoT vision solutions, this model provides production-ready detection and segmentation with minimal setup.

RF-DETR is a real-time transformer-based object detection and instance segmentation architecture developed by Roboflow, achieving state-of-the-art results on COCO (presented at ICLR 2026). It builds on a **DINOv2 ViT backbone** with a DETR-style decoder and hierarchical feature pyramid, offering six size variants from Nano to 2XLarge for flexible accuracy-speed trade-offs. Detection variants (Nano through Large) are released under Apache 2.0; XLarge and 2XLarge variants are available under the PML 1.0 license via `rfdetr[plus]`.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

### Detection Models

| Dataset | Model Name        | Model ID    | Input Size | mAP[.5:.95]% | mAP[.50]% | Available | Notes                    |
| -       | -                 | -           | -          | -            | -         | -         | -                        |
| COCO    | rfdetr-nano       | `od-mhTBD`  | 384×384    | 48.4         | 67.6      | ✅        | Recommended, Smallest    |
| COCO    | rfdetr-small      | `od-mhTBD`  | 512×512    | 53.0         | 72.1      | ✅        |                          |
| COCO    | rfdetr-medium     | `od-mhTBD`  | 576×576    | 54.7         | 73.6      | ✅        |                          |
| COCO    | rfdetr-large      | `od-mhTBD`  | 704×704    | 56.5         | 75.1      | ✅        |                          |
| COCO    | rfdetr-xlarge     | `od-mhTBD`  | 700×700    | 58.6         | 77.4      | ✅        | Requires `--plus`; PML 1.0 |
| COCO    | rfdetr-2xlarge    | `od-mhTBD`  | 880×880    | 60.1         | 78.5      | ✅        | Requires `--plus`; PML 1.0 |

### Segmentation Models

| Dataset | Model Name            | Model ID    | Input Size | mAP[.5:.95]% | mAP[.50]% | Available | Notes                    |
| -       | -                     | -           | -          | -            | -         | -         | -                        |
| COCO    | rfdetr-seg-nano       | `od-mhTBD`  | 312×312    | 40.3         | 63.0      | ✅        | Recommended, Smallest    |
| COCO    | rfdetr-seg-small      | `od-mhTBD`  | 384×384    | 43.1         | 66.2      | ✅        |                          |
| COCO    | rfdetr-seg-medium     | `od-mhTBD`  | 432×432    | 45.3         | 68.4      | ✅        |                          |
| COCO    | rfdetr-seg-large      | `od-mhTBD`  | 504×504    | 47.1         | 70.5      | ✅        |                          |
| COCO    | rfdetr-seg-xlarge     | `od-mhTBD`  | 624×624    | 48.8         | 72.2      | ✅        | Requires `--plus`; PML 1.0 |
| COCO    | rfdetr-seg-2xlarge    | `od-mhTBD`  | 768×768    | 49.9         | 73.1      | ✅        | Requires `--plus`; PML 1.0 |

> Latency benchmarks measured on NVIDIA T4 GPU with TensorRT FP16. mAP values are on COCO val2017.

---

## Setup & Installation

### Requirements
```bash
# Activate your Python virtual environment, then install dependencies.
# ONNX export dependencies (auto-installed by the download script)
pip install "rfdetr[onnx]"

# For XLarge / 2XLarge variants (PML 1.0 license)
pip install "rfdetr[onnx,plus]"

# Inference and deployment
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
```

#### For TI hardware deployment
Refer the link for **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)**

---

## Usage

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Export

Pretrained COCO weights are downloaded automatically from HuggingFace on first use. Use `download_rfdetr.py` to export any variant to ONNX.

```bash
# Activate your virtual environment, then:

# List all available variants with accuracy and latency info
python download_rfdetr.py --list-models

# Export the default model (rfdetr_nano)
python download_rfdetr.py

# Export a specific variant
python download_rfdetr.py --model rfdetr_nano
python download_rfdetr.py --model rfdetr_small
python download_rfdetr.py --model rfdetr_medium
python download_rfdetr.py --model rfdetr_large

# Export multiple variants at once
python download_rfdetr.py --model rfdetr_nano rfdetr_small rfdetr_medium rfdetr_large

# Export XLarge / 2XLarge (requires rfdetr[plus], PML 1.0 license)
python download_rfdetr.py --model rfdetr_xlarge rfdetr_2xlarge --plus

# Export segmentation variants
python download_rfdetr.py --model rfdetr_seg_nano rfdetr_seg_small rfdetr_seg_medium rfdetr_seg_large

# Export with a custom input resolution
python download_rfdetr.py --model rfdetr_medium --shape 608 608

# Export backbone feature extractor only
python download_rfdetr.py --model rfdetr_nano --backbone-only

# Export from a locally trained checkpoint
python download_rfdetr.py --model rfdetr_medium --weights /path/to/custom.pth

# Export with a specific ONNX opset version
python download_rfdetr.py --model rfdetr_nano --opset 18

# Save to a custom output directory
python download_rfdetr.py --model rfdetr_nano --output-dir ./exports
```

The script will automatically:
1. Install `rfdetr[onnx]` (or `rfdetr[onnx,plus]` for XLarge/2XLarge) if not present
2. Download pretrained COCO weights from HuggingFace on first use
3. Export to ONNX with static batch dimension
4. Save as `rfdetr_<variant>.onnx` in the output directory

#### Export script parameters

| Flag             | Default              | Description                                              |
| -                | -                    | -                                                        |
| `--model`        | `rfdetr_nano`        | One or more variant names. See `--list-models`.          |
| `--shape H W`    | Model default        | Custom input resolution (height width).                  |
| `--opset`        | `17`                 | ONNX opset version.                                      |
| `--batch-size`   | `1`                  | Batch size embedded in the exported graph.               |
| `--backbone-only`| `False`              | Export DINOv2 backbone only.                             |
| `--weights`      | COCO pretrained      | Path to a local `.pth` checkpoint.                       |
| `--output-dir`   | Script directory     | Directory where `.onnx` files are saved.                 |
| `--plus`         | `False`              | Enable XLarge/2XLarge export (requires PML 1.0 consent). |
| `--quiet`        | `False`              | Suppress rfdetr's internal verbose output.               |
| `--list-models`  | —                    | Print model catalogue table and exit.                    |

### Using the model on TI device

#### Option 1: Advanced Users (TIDL Tools)
For users familiar with TIDL and requiring fine-grained control:

```bash
# Use edgeai-tidl-tools for compilation and inference
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tools
```

Refer to the [tidl-tools setup](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/README.md#getting-started) page for more details on compile and infer.

**Learn more:** [edgeai-tidl-tools documentation](https://github.com/TexasInstruments/edgeai-tidl-tools)

#### Option 2: Simplified Workflow (Recommended)
For easy compilation, benchmarking, and accuracy evaluation:

Setup tidl runner using this link: [edgeai-tidlrunner setup](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tidlrunner/docs/setup.md)

```bash
# Compile model and evaluate performance on J784S4.

# rfdetr-nano (384×384)
tidlrunner-cli compile --target_device J784S4 --config_path rfdetr_nano_config.yaml

# rfdetr-small (512×512)
tidlrunner-cli compile --target_device J784S4 --config_path rfdetr_small_config.yaml

# rfdetr-medium (576×576)
tidlrunner-cli compile --target_device J784S4 --config_path rfdetr_medium_config.yaml

# rfdetr-large (704×704)
tidlrunner-cli compile --target_device J784S4 --config_path rfdetr_large_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

---

## Citation

If you use RF-DETR in your research, please cite:

```bibtex
@inproceedings{rfdetr2026,
  title     = {RF-DETR: Real-Time Detection Transformer},
  author    = {Roboflow},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://github.com/roboflow/rf-detr}
}
```

---

## Additional Resources

### Tools & Frameworks
- [EdgeAI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - Low-level compilation and inference
- [EdgeAI TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner) - High-level wrapper for easy deployment
- [EdgeAI Main Repository](https://github.com/TexasInstruments/edgeai) - Complete EdgeAI ecosystem

### RF-DETR Resources
- [RF-DETR Official Repository](https://github.com/roboflow/rf-detr) - Roboflow implementation
- [RF-DETR Documentation](https://rfdetr.roboflow.com) - Full API and usage docs
- [RF-DETR on HuggingFace](https://huggingface.co/roboflow) - Pretrained weight hub

**License:** apache-2.0 (Nano–Large); PML 1.0 (XLarge, 2XLarge)  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** July 2026
