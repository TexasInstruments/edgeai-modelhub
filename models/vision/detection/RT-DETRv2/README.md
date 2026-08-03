---
license: apache-2.0
tags:
- vision
- image-detection
datasets:
- COCO
---

# RT-DETRv2 for TI EdgeAI

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

🚀 **Ready-to-deploy real-time object detection for TI edge devices**

This RT-DETRv2 model is specifically optimized for **Texas Instruments MPU (Microprocessor Unit) devices**, enabling high-performance computer vision applications at the edge.

RT-DETRv2 (Real-Time Detection Transformer v2) is the improved version of RT-DETR, presented at **CVPR 2024**. It is a real-time, end-to-end object detection transformer that eliminates the need for hand-crafted anchor boxes and NMS post-processing. Built on a **ResNet-vd hybrid encoder** backbone with a transformer decoder, it achieves state-of-the-art accuracy–speed trade-offs on COCO across five size variants (S, M*, M, L, X).

All variants are released under the Apache 2.0 license.

### Learn More About TI EdgeAI Platform

- 📖 **[EdgeAI SDK Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)** - Complete SDK guide, installation, and system setup
- 🔧 **[EdgeAI MPU Overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)** - Architecture details, performance benchmarks, and development resources
- 🌐 **[TI EdgeAI Ecosystem](https://github.com/TexasInstruments/edgeai)** - Explore the full EdgeAI toolkit and model zoo

---

## Model Details

| Dataset | Model Name     | Model ID       | Input Size | mAP[.5:.95]% | mAP[.50]% | Params (M) | FLOPs (G) | Available | Notes                 |
| -       | -              | -              | -          | -            | -         | -          | -         | -         | -                     |
| COCO    | rtdetrv2-s     | `od-mhTBD`     | 640×640    | 48.1         | 65.1      | 20         | 60        | ✅        | Recommended, Smallest |
| COCO    | rtdetrv2-ms    | `od-mhTBD`     | 640×640    | 49.9         | 67.5      | 31         | 92        | ✅        | M* lighter variant    |
| COCO    | rtdetrv2-m     | `od-mhTBD`     | 640×640    | 51.9         | 69.9      | 36         | 100       | ✅        |                       |
| COCO    | rtdetrv2-l     | `od-mhTBD`     | 640×640    | 53.4         | 71.6      | 42         | 136       | ✅        |                       |
| COCO    | rtdetrv2-x     | `od-mhTBD`     | 640×640    | 54.3         | 72.8      | 76         | 259       | ✅        | Largest               |

> mAP evaluated on COCO val2017. FPS measured on NVIDIA T4 GPU with TensorRT FP16, batch=1.

### ONNX Output Format

Each exported model produces two outputs (batch=1 by default):

| Output Name   | Shape          | Description                                         |
| -             | -              | -                                                   |
| `pred_boxes`  | `[1, 300, 4]`  | CxCyWH normalised [0,1] – per-query box predictions |
| `pred_logits` | `[1, 300, 80]` | Raw class logits (apply sigmoid for probabilities)  |

Apply `sigmoid` to `pred_logits` to get class scores. Boxes are in CxCyWH [0,1] format relative to input image size.

---

## Setup & Installation

### Requirements
```bash
# Activate your Python virtual environment, then:

# Core dependencies (auto-installed by prepare_model.py)
pip install torch>=2.0.1 torchvision>=0.15.2 scipy PyYAML onnx

# Optional: ONNX simplifier
pip install onnxsim

# Inference and deployment
pip install onnxruntime>=1.16.0
```

The `prepare_model.py` script automatically clones the RT-DETR source repository
(to `~/.cache/rtdetr_src`) and downloads pretrained COCO weights from GitHub
Releases on first use.

#### For TI hardware deployment
Refer to **[tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)** for device compilation.

---

## Usage

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### Model Export

Pretrained COCO weights are downloaded automatically on first use. Use `prepare_model.py` to export any variant to ONNX.

```bash
# Activate your virtual environment, then:

# List all available variants with accuracy and latency info
python prepare_model.py --list-models

# Export the default model (rtdetrv2_s)
python prepare_model.py

# Export a specific variant
python prepare_model.py --model rtdetrv2_s
python prepare_model.py --model rtdetrv2_ms
python prepare_model.py --model rtdetrv2_m
python prepare_model.py --model rtdetrv2_l
python prepare_model.py --model rtdetrv2_x

# Export multiple variants at once
python prepare_model.py --model rtdetrv2_s rtdetrv2_m rtdetrv2_l

# Export with a custom input resolution
python prepare_model.py --model rtdetrv2_l --shape 800 800

# Export from a locally trained checkpoint
python prepare_model.py --model rtdetrv2_m --weights /path/to/custom.pth

# Export with ONNX simplification applied
python prepare_model.py --model rtdetrv2_s --simplify

# Export with a specific ONNX opset version
python prepare_model.py --model rtdetrv2_s --opset 18

# Save to a custom output directory
python prepare_model.py --model rtdetrv2_s --output-dir ./exports
```

The script will automatically:
1. Install core Python dependencies (`torch`, `scipy`, `PyYAML`, `onnx`) if missing
2. Clone the RT-DETR source repository to `~/.cache/rtdetr_src` if not present
3. Download pretrained COCO weights from GitHub Releases on first use
4. Export to ONNX with two outputs: `pred_boxes` and `pred_logits`
5. Run ONNX shape inference

#### Export script parameters

| Flag             | Default              | Description                                                |
| -                | -                    | -                                                          |
| `--model`        | `rtdetrv2_s`         | One or more variant names. See `--list-models`.            |
| `--shape H W`    | `640 640`            | Custom input resolution (height width).                    |
| `--opset`        | `16`                 | ONNX opset version.                                        |
| `--batch-size`   | `1`                  | Batch size embedded in the exported graph.                 |
| `--weights`      | COCO pretrained      | Path to a local `.pth` checkpoint.                         |
| `--output-dir`   | Script directory     | Directory where `.onnx` files are saved.                   |
| `--force`        | `False`              | Re-export even if the `.onnx` already exists.              |
| `--simplify`     | `False`              | Apply onnxsim after export (requires `pip install onnxsim`).|
| `--quiet`        | `False`              | Suppress verbose loading output.                           |
| `--list-models`  | —                    | Print model catalogue table and exit.                      |

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

# rtdetrv2-s (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtdetrv2_s_config.yaml

# rtdetrv2-ms (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtdetrv2_ms_config.yaml

# rtdetrv2-m (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtdetrv2_m_config.yaml

# rtdetrv2-l (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtdetrv2_l_config.yaml

# rtdetrv2-x (640×640)
tidlrunner-cli compile --target_device J784S4 --config_path rtdetrv2_x_config.yaml
```

To evaluate accuracy, replace `compile` with `evaluate` in the commands above.

**Learn more:** [edgeai-tidlrunner documentation](https://github.com/TexasInstruments/edgeai-tidlrunner)

---

## Citation

If you use RT-DETRv2 in your research, please cite:

```bibtex
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
  title     = {RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
  author    = {Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
  year      = {2024},
  eprint    = {2407.17140},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2407.17140}
}

@misc{lv2023detrs,
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  author    = {Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei
               and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
  year      = {2023},
  eprint    = {2304.08069},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

---

## Additional Resources

### Tools & Frameworks
- [EdgeAI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - Low-level compilation and inference
- [EdgeAI TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner) - High-level wrapper for easy deployment
- [EdgeAI Main Repository](https://github.com/TexasInstruments/edgeai) - Complete EdgeAI ecosystem

### RT-DETRv2 Resources
- [RT-DETR Official Repository](https://github.com/lyuwenyu/RT-DETR) - Official PyTorch and PaddlePaddle implementations
- [RT-DETRv2 Paper](https://arxiv.org/abs/2407.17140) - arXiv preprint
- [RT-DETR CVPR 2024 Paper](https://arxiv.org/abs/2304.08069) - Original RT-DETR paper

**License:** Apache 2.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** July 2026
