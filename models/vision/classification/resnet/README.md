---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
---

<div align="center">

# 🖼️ ResNet-50 for TI EdgeAI

### Deep Residual Network for Image Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-ONNX-orange?style=for-the-badge)](https://onnx.ai/)
[![Task](https://img.shields.io/badge/Task-Classification-green?style=for-the-badge)](https://github.com/TexasInstruments/edgeai)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet--1K-blueviolet?style=for-the-badge)](http://www.image-net.org/)

</div>

---

## 📋 Overview

**ResNet-50** is a 50-layer deep convolutional neural network optimized for **Texas Instruments MPU devices**. This folder provides two production-ready ONNX variants that deliver industry-leading accuracy on ImageNet classification while maintaining efficient computation suitable for edge deployment.

✅ **25.6M Parameters** - Efficient architecture  
✅ **76.15% Top-1 Accuracy** - ImageNet benchmark  
✅ **4.1 GigaMACs** - Low computational cost  
✅ **Hardware-Optimized** - TI J784S4 validated  

---

## 📊 Model Variants

| Property | resNet50 | resnet50-v1-7 |
|----------|----------|---------------|
| **Model ID** | `cl-mh6000` | `cl-mh6001` |
| **ONNX File** | `resnet50.onnx` | `resnet50-v1-7.onnx` |
| **Config File** | `resnet50_config.yaml` | `resnet50-v1-7_config.yaml` |
| **Input Size** | 224×224 RGB | 224×224 RGB |
| **Parameters** | ~25.6M | ~25.6M |
| **Computation** | 4.1 GigaMACs | 4.1 GigaMACs |
| **Top-1 Accuracy** | 76.15% | 76.15% |
| **Framework** | ONNX | ONNX |
| **License** | Apache 2.0 | Apache 2.0 |
| **Training Dataset** | ImageNet-1K | ImageNet-1K |
| **Source** | [onnx-community/resnet-50-ONNX](https://huggingface.co/onnx-community/resnet-50-ONNX) | [onnxmodelzoo/resnet50-v1-7](https://huggingface.co/onnxmodelzoo/resnet50-v1-7) |

---

## 🚀 Quick Start

### Download from HuggingFace
If you are accessing this from HuggingFace, clone the repository using the `hf` CLI:

```bash
hf download <REPO_ID> --local-dir <download_location>
```

### 1️⃣ Prerequisites

```bash
pip install onnx>=1.22.0 onnxruntime>=1.23.2
```

### 2️⃣ Download Models

```bash
# Download resNet50 (cl-mh6000)
python prepare_model.py --link-file resnet50.onnx.link

# Download resnet50-v1-7 (cl-mh6001)
python prepare_model.py --link-file resnet50-v1-7.onnx.link
```

The script automatically:
- 📥 Downloads from HuggingFace
- 🔧 Fixes dynamic shapes to static
- ✅ Validates model structure

### 3️⃣ Compile for TI Hardware

**Using TIDL Runner (Recommended):**

```bash
# resNet50
tidlrunner-cli compile --target_device J784S4 \
  --config_path resnet50_config.yaml

# resnet50-v1-7
tidlrunner-cli compile --target_device J784S4 \
  --config_path resnet50-v1-7_config.yaml
```

**Using TIDL Tools (Advanced):**

```bash
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tools
# Follow setup at https://github.com/TexasInstruments/edgeai-tidl-tools
```

### 4️⃣ Evaluate Performance

```bash
# resNet50
tidlrunner-cli evaluate --target_device J784S4 \
  --config_path resnet50_config.yaml

# resnet50-v1-7
tidlrunner-cli evaluate --target_device J784S4 \
  --config_path resnet50-v1-7_config.yaml
```

---

## 🛠️ Setup & Installation

### Requirements

```bash
# Python dependencies
pip install onnx>=1.22.0
pip install onnxruntime>=1.23.2
```

### Hardware Deployment

- **Primary Target:** TI J784S4 MPU
- **Framework:** TIDL Compilation Tools
- **Deployment:** [TIDL Runner](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/README.md)

---

## 📚 Deployment Options

<table>
<tr>
<td width="50%">

### 🎯 Recommended: TIDL Runner
High-level interface for easy deployment

**Best for:**
- Quick prototyping
- Model evaluation
- Benchmark testing

[Setup Guide →](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tidlrunner/docs/setup.md)

</td>
<td width="50%">

### ⚙️ Advanced: TIDL Tools
Low-level compilation framework

**Best for:**
- Custom optimization
- Fine-grained control
- Production deployment

[Documentation →](https://github.com/TexasInstruments/edgeai-tidl-tools)

</td>
</tr>
</table>

---

## 💡 Use Cases

| Application | Details |
|-------------|---------|
| 🏭 **Industrial** | Product classification, defect detection |
| 📹 **Surveillance** | Scene understanding, object categorization |
| 🤖 **Robotics** | Visual perception, environment understanding |
| 🌐 **IoT** | Edge inference, smart cameras |

---

## 📖 Citation

If you use this model, please cite:

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision 
            and pattern recognition},
  pages={770--778},
  year={2016}
}
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| 📖 **Paper** | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 🔧 **TIDL Tools** | [GitHub](https://github.com/TexasInstruments/edgeai-tidl-tools) |
| 🚀 **TIDL Runner** | [GitHub](https://github.com/TexasInstruments/edgeai-tidlrunner) |
| 🌐 **EdgeAI SDK** | [Documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) |
| 📦 **ONNX Format** | [onnx.ai](https://onnx.ai/) |

---

## 🔀 Related Models

<table>
<tr>
<td align="center">

**ResNet-18**  
Lightweight variant  
Lower accuracy

</td>
<td align="center">

**ResNet-101**  
Deeper variant  
Higher accuracy

</td>
<td align="center">

**MobileNetV2**  
Mobile-optimized  
Ultra-lightweight

</td>
<td align="center">

**EfficientNet**  
Efficiency-focused  
Flexible scaling

</td>
</tr>
</table>

---

<div align="center">

**License:** Apache 2.0  
**Maintained by:** Texas Instruments EdgeAI Team  
**Last Updated:** August 2026

[🏠 Back to Model Hub](../../README.md)

</div>
