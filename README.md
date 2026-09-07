<div align="center">

<img src="docs/assets/TXN-Logo.png" alt="Texas Instruments" width="72" height="72" />

# EdgeAI Model Hub

**Pre-trained, hardware-optimized AI models for TI edge devices**

</div>

---

## Overview

The TI EdgeAI Model Hub is a curated repository of open-source computer vision models
optimized for deployment on Texas Instruments MPU devices. Models are compiled for TI hardware
using [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)
or [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner),
enabling production-ready inference without cloud dependency. For more details on TIDL model
compilation options, runtimes, and supported operators, see the
[TIDL User Guide](https://github.com/TexasInstruments/edgeai-tidl-tools#user-guide).

- ✅ Optimized for TI MPU processors
- ✅ Benchmarked on real TI hardware
- ✅ Portable across all various TI MPU devices
- ✅ Automated scripts for model compilation, benchmark & deployment

## Use Cases

| | | |
|---|---|---|
| **Automotive** | **Aerospace & Defense** | **Industrial** |
| **Surveillance** | **Robotics** | **Edge IoT** |

## License Summary

Models in this hub are distributed under various open-source licenses — each model's license is indicated in its own documentation page.

> **Disclaimer:** Certain licenses in this repository impose distribution restrictions that
> may affect commercial, proprietary, or regulated-industry use. It is the sole responsibility of the
> user to review the applicable license terms, assess compatibility with their intended use, and obtain
> any necessary legal clearances prior to use or distribution. Texas Instruments makes no representation
> regarding the suitability of these licenses for any particular purpose and accepts no legal
> responsibility for the user's compliance obligations.

## Compilation & Deployment

| Tool | Description |
|------|-------------|
| **[edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner)** (Recommended) | High-level compilation and benchmark interface. |
| **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)** | Deployment tools (and also low-level compilation tools for advanced users). |

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/TexasInstruments/edgeai-modelhub.git
cd edgeai-modelhub
```

**2. Navigate to a model directory and prepare the model**
```bash
cd models/vision/<task>/<model>/
python prepare_model.py --model <variant>
```

**3. Compile for TI hardware** (run from inside the edgeai-tidlrunner directory)
```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device <device> \
  --config_path /path/to/edgeai-modelhub/<model>_config.yaml
```

**4. Infer on TI hardware** (run from inside the edgeai-tidlrunner directory)
```bash
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device <device> \
  --config_path /path/to/edgeai-modelhub/<model>_config.yaml
```

## Deployment
Deplyment can be done using **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)**. For ONNX models, onnxruntime-tidl with TIDL acceleration can be used. Consult the documentation of edgeai-tidl-tools for more details.


## Supported Hardware

Compatible TI MPU device families compiled and validated via TIDL. See the supported devices, SDKs and version compatibility at the
[EdgeAI developer landing space](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) and the
[edgeai-tidl-tools SDK version compatibility matrix](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/sdk_version_compatibility_table.md).

| Device Family | Variants |
|---|---|
| **AM62A** | [AM62A3](https://www.ti.com/product/AM62A3) · [AM62A7](https://www.ti.com/product/AM62A7) |
| **J722S** | [TDA4AEN](https://www.ti.com/product/TDA4AEN-Q1) · [AM67A](https://www.ti.com/product/AM67A) |
| **J721E** | [TDA4VM](https://www.ti.com/product/TDA4VM) |
| **J721S2** | [TDA4VE](https://www.ti.com/product/TDA4VE-Q1) · [TDA4VL](https://www.ti.com/product/TDA4VL-Q1) · [TDA4AL](https://www.ti.com/product/TDA4AL-Q1) · [AM68A](https://www.ti.com/product/AM68A) |
| **J784S4** | [TDA4VH](https://www.ti.com/product/TDA4VH-Q1) · [TDA4AH](https://www.ti.com/product/TDA4AH-Q1) · [AM69A](https://www.ti.com/product/AM69A) |

## Model Catalog

| Model | Capability | Variants | Input | Performance | License | Docs |
|---|---|---|---|---|---|---|
| **MobileNetV3** | Image Classification | large | 224×224 | 75.3% Top-1 | [![BSD-3-Clause](https://img.shields.io/badge/BSD--3--Clause-065f46?style=flat-square)](https://opensource.org/licenses/BSD-3-Clause) | [View](models/vision/classification/MobileNetV3/) |
| **ResNet-50** | Image Classification | v1.5, v1 | 224×224 | 74.93–76.15% Top-1 | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/classification/ResNet/) |
| **DINO** | Image Classification | ViT-S/16, ViT-S/8, ViT-B/16, ViT-B/8, ResNet-50 | 224×224 | 75.3–80.1% Top-1 | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/classification/DINO/) |
| **DINOv2** | Image Classification | ViT-S/14, ViT-B/14 (w/ & w/o registers) | 224×224 | 80.9–84.6% Top-1 | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/classification/DINOv2/) |
| **ViT** | Image Classification | vit_b_16, vit_b_32, vit_l_16, vit_l_32 | 224×224 | 75.9–81.1% Top-1 | [![BSD-3-Clause](https://img.shields.io/badge/BSD--3--Clause-065f46?style=flat-square)](https://opensource.org/licenses/BSD-3-Clause) | [View](models/vision/classification/ViT/) |
| **ConvNeXt** | Image Classification | convnext_tiny, convnext_small, convnext_base, convnext_large | 224×224 | 82.5–84.4% Top-1 | [![BSD-3-Clause](https://img.shields.io/badge/BSD--3--Clause-065f46?style=flat-square)](https://opensource.org/licenses/BSD-3-Clause) | [View](models/vision/classification/ConvNeXt/) |
| **DEIMv2** | Object Detection | s, m | 640×640 | 50.9–53.0% mAP | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/DEIMv2/) |
| **DETR** | Object Detection | detr_resnet50, detr_resnet50_dc5, detr_resnet101, detr_resnet101_dc5 | 800×800 (flexible) | AP50:95 42.0–44.9, AP50 62.4–64.7 | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/DETR/) |
| **Deformable-DETR** | Object Detection | single-scale | 800×800 | AP50:95 39.4% | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/Deformable-DETR/) |
| **RF-DETR** | Object Detection | nano, s, m, l | 384–704px | 48.4–56.5% mAP | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/RF-DETR/) |
| **RT-DETRv2** | Object Detection | s, ms, m, l, x | 640×640 | 48.1–54.3% mAP | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/RT-DETRv2/) |
| **RTMDet** | Object Detection | tiny, s, m, l, x | 640×640 | 40.9–52.8% mAP | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/RTMDet/) |
| **YOLO11** | Object Detection | n, s, m, l, x | 640×640 | 39.5–54.7% mAP | [![AGPL 3.0](https://img.shields.io/badge/AGPL%203.0-9d174d?style=flat-square)](https://www.gnu.org/licenses/agpl-3.0.html) | [View](models/vision/detection/YOLO11/) |
| **YOLO26** | Object Detection | n, s, m, l, x | 640×640 | 40.9–57.5% mAP | [![AGPL 3.0](https://img.shields.io/badge/AGPL%203.0-9d174d?style=flat-square)](https://www.gnu.org/licenses/agpl-3.0.html) | [View](models/vision/detection/YOLO26/) |
| **YOLOv8** | Object Detection | n, m | 640×640 | 37.3–50.2% mAP | [![AGPL 3.0](https://img.shields.io/badge/AGPL%203.0-9d174d?style=flat-square)](https://www.gnu.org/licenses/agpl-3.0.html) | [View](models/vision/detection/YOLOv8/) |
| **YOLOX** | Object Detection | nano, tiny, m, l, x, darknet53 | 416×416 / 640×640 | 24.8–51.2% mAP | [![Apache 2.0](https://img.shields.io/badge/Apache%202.0-1d4ed8?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0) | [View](models/vision/detection/YOLOX/) |

## Resources & Links

- **Ecosystem:** [TI EdgeAI](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu) · [EdgeAI SDK](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)
- **Tools:** [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) · [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)
- **Community:** [E2E Support](https://e2e.ti.com/support/processors-group/processors/f/processors-forum) · [Issues](https://github.com/TexasInstruments/edgeai/issues) · [Discussions](https://github.com/TexasInstruments/edgeai/discussions)

---

<div align="center">

Maintained by Texas Instruments EdgeAI Team &nbsp;|&nbsp; Last Updated August 2026

[Contact](mailto:edgeai-dev@list.ti.com)

</div>
