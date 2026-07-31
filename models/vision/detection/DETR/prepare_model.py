"""Script to export DETR pretrained ONNX model(s).

DETR (Detection TRansformer) from Facebook Research.
Models are loaded via torch.hub, which automatically clones the DETR repository
and downloads pretrained COCO weights from dl.fbaipublicfiles.com on first use.

Reference: https://github.com/facebookresearch/detr

Detection variants (Apache 2.0, COCO pretrained):
  detr_resnet50       – 800×800, ~41M params,  AP50:95 42.0,  AP50 62.4
  detr_resnet50_dc5   – 800×800, ~41M params,  AP50:95 43.3,  AP50 63.1
  detr_resnet101      – 800×800, ~60M params,  AP50:95 43.5,  AP50 63.8
  detr_resnet101_dc5  – 800×800, ~60M params,  AP50:95 44.9,  AP50 64.7

Panoptic segmentation variants (Apache 2.0, COCO pretrained):
  detr_resnet50_panoptic      – 800×800, ~43M params,  PQ 43.4  (box AP 38.8)
  detr_resnet50_dc5_panoptic  – 800×800, ~43M params,  PQ 44.6  (box AP 40.2)
  detr_resnet101_panoptic     – 800×800, ~62M params,  PQ 45.1  (box AP 40.1)

DC5 = dilated convolutions in ResNet's last block (stride 16→32 → stride 8→16),
      yielding higher-resolution feature maps at the cost of increased computation.

ONNX inputs/outputs:
  Input  : images       – (N, 3, H, W) float32, ImageNet-normalized
  Output : pred_boxes   – (N, 100, 4)   boxes in (cx, cy, w, h), normalized [0, 1]
           pred_logits  – (N, 100, 92)  class logits (det) or (N, 100, 251) (panoptic)
           pred_masks   – (N, 100, H/4, W/4) panoptic mask logits (panoptic only)

Notes:
  - DETR always outputs exactly 100 query slots per image.
  - Post-processing: apply softmax over pred_logits and filter out slots where the
    no-object class (index 91 for detection, 250 for panoptic) has the highest score.
  - DETR trains with variable-size inputs (shorter-side 800, max 1333). For ONNX a
    fixed square shape is used (default 800×800). Any size works; 800px gives best AP.
  - First run requires internet access to clone the DETR repo and download weights.

Usage:
  python prepare_model.py
  python prepare_model.py --model detr_resnet50
  python prepare_model.py --model detr_resnet50 detr_resnet101
  python prepare_model.py --model detr_resnet50 --shape 800 1333
  python prepare_model.py --model detr_resnet50 --weights /path/to/checkpoint.pth
  python prepare_model.py --model detr_resnet50 --opset 18 --output-dir ./exports
  python prepare_model.py --list-models
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys


# ─────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────

# Each entry: variant_key → metadata dict
#   hub_name   : function name used with torch.hub.load
#   task       : "detection" or "panoptic"
#   num_classes: 91 for detection (outputs 92 logits incl. no-object),
#                250 for panoptic (outputs 251 logits incl. no-object)
MODEL_CATALOG: dict[str, dict] = {
    # ── Detection ─────────────────────────────────────────────────────────────
    "detr_resnet50": {
        "hub_name":     "detr_resnet50",
        "shape":        (800, 800),
        "params_m":     41.3,
        "ap50_95":      42.0,
        "ap50":         62.4,
        "pq":           None,
        "latency_ms":   36.0,
        "license":      "Apache 2.0",
        "task":         "detection",
        "num_classes":  91,
        "backbone":     "ResNet-50",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
    },
    "detr_resnet50_dc5": {
        "hub_name":     "detr_resnet50_dc5",
        "shape":        (800, 800),
        "params_m":     41.3,
        "ap50_95":      43.3,
        "ap50":         63.1,
        "pq":           None,
        "latency_ms":   83.0,
        "license":      "Apache 2.0",
        "task":         "detection",
        "num_classes":  91,
        "backbone":     "ResNet-50 DC5",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth",
    },
    "detr_resnet101": {
        "hub_name":     "detr_resnet101",
        "shape":        (800, 800),
        "params_m":     60.0,
        "ap50_95":      43.5,
        "ap50":         63.8,
        "pq":           None,
        "latency_ms":   50.0,
        "license":      "Apache 2.0",
        "task":         "detection",
        "num_classes":  91,
        "backbone":     "ResNet-101",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",
    },
    "detr_resnet101_dc5": {
        "hub_name":     "detr_resnet101_dc5",
        "shape":        (800, 800),
        "params_m":     60.0,
        "ap50_95":      44.9,
        "ap50":         64.7,
        "pq":           None,
        "latency_ms":   97.0,
        "license":      "Apache 2.0",
        "task":         "detection",
        "num_classes":  91,
        "backbone":     "ResNet-101 DC5",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth",
    },
    # ── Panoptic segmentation ──────────────────────────────────────────────────
    "detr_resnet50_panoptic": {
        "hub_name":     "detr_resnet50_panoptic",
        "shape":        (800, 800),
        "params_m":     43.2,
        "ap50_95":      38.8,
        "ap50":         None,
        "pq":           43.4,
        "latency_ms":   None,
        "license":      "Apache 2.0",
        "task":         "panoptic",
        "num_classes":  250,
        "backbone":     "ResNet-50",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth",
    },
    "detr_resnet50_dc5_panoptic": {
        "hub_name":     "detr_resnet50_dc5_panoptic",
        "shape":        (800, 800),
        "params_m":     43.2,
        "ap50_95":      40.2,
        "ap50":         None,
        "pq":           44.6,
        "latency_ms":   None,
        "license":      "Apache 2.0",
        "task":         "panoptic",
        "num_classes":  250,
        "backbone":     "ResNet-50 DC5",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth",
    },
    "detr_resnet101_panoptic": {
        "hub_name":     "detr_resnet101_panoptic",
        "shape":        (800, 800),
        "params_m":     62.0,
        "ap50_95":      40.1,
        "ap50":         None,
        "pq":           45.1,
        "latency_ms":   None,
        "license":      "Apache 2.0",
        "task":         "panoptic",
        "num_classes":  250,
        "backbone":     "ResNet-101",
        "pth_url":      "https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth",
    },
}

DEFAULT_MODEL = "detr_resnet50"

# torch.hub repo string for DETR
_HUB_REPO = "facebookresearch/detr:main"


# ─────────────────────────────────────────────
# Dependency management
# ─────────────────────────────────────────────

def _pip_install(*packages: str) -> None:
    """Install *packages* via pip, suppressing verbose output."""
    print(f"[DEP] Installing: {', '.join(packages)} …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"[DEP] ERROR: pip install failed (exit code {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip())
        print("[DEP] Please install manually and re-run:")
        print(f"      pip install {' '.join(packages)}")
        sys.exit(1)
    print("[DEP] Installation complete.\n")


def ensure_dependencies() -> None:
    """Ensure torch, torchvision, onnx, and scipy are importable.

    scipy is required because DETR's model code imports it at module load
    time (scipy.optimize.linear_sum_assignment in models/matcher.py).
    """
    required = [
        ("torch",       "torch>=1.12.0"),
        ("torchvision", "torchvision>=0.13.0"),
        ("onnx",        "onnx>=1.14.0"),
        ("scipy",       "scipy"),
    ]
    missing_pip = []
    for mod_name, pip_spec in required:
        try:
            importlib.import_module(mod_name)
            print(f"[DEP]  ✔  {mod_name} is installed.")
        except ImportError:
            print(f"[DEP]  ✘  {mod_name} not found.")
            missing_pip.append(pip_spec)

    if missing_pip:
        _pip_install(*missing_pip)

    print()


# ─────────────────────────────────────────────
# Hub path helpers
# ─────────────────────────────────────────────

def _add_detr_to_path() -> str:
    """Add the downloaded DETR source directory to sys.path (index 0).

    torch.hub.load clones facebookresearch/detr to
    ``<hub_dir>/facebookresearch_detr_main/``.  This directory must be on
    sys.path so that ``from util.misc import NestedTensor`` succeeds when
    building the ONNX wrapper.

    Returns the DETR root directory path.
    """
    import torch.hub as hub

    hub_dir = hub.get_dir()
    if not os.path.isdir(hub_dir):
        raise RuntimeError(
            f"torch.hub directory not found: {hub_dir}. "
            "Run the script with internet access so torch.hub can clone DETR."
        )

    for entry in sorted(os.listdir(hub_dir), reverse=True):
        if entry.startswith("facebookresearch_detr"):
            detr_root = os.path.join(hub_dir, entry)
            if os.path.isdir(detr_root):
                if detr_root not in sys.path:
                    sys.path.insert(0, detr_root)
                return detr_root

    raise RuntimeError(
        "Could not find DETR source in torch hub directory.\n"
        f"Expected a subdirectory starting with 'facebookresearch_detr' inside {hub_dir}.\n"
        "This is populated automatically by torch.hub.load on first use."
    )


# ─────────────────────────────────────────────
# ONNX export wrappers
# ─────────────────────────────────────────────

def _make_wrapper(model, NestedTensor, task: str):
    """Return an nn.Module that accepts a plain image tensor and produces flat outputs.

    DETR's forward pass expects a NestedTensor (image + padding mask).  These
    wrappers create a zero mask (no padding) for fixed-size ONNX export, making
    the model accept a standard (N, 3, H, W) float32 tensor.

    Output order:
      detection : pred_boxes (N,100,4), pred_logits (N,100,92)
      panoptic  : pred_boxes (N,100,4), pred_logits (N,100,251), pred_masks (N,100,H/4,W/4)
    """
    import torch
    import torch.nn as nn

    if task == "detection":
        class _DetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self._NT = NestedTensor

            def forward(self, images: torch.Tensor):
                B, _, H, W = images.shape
                mask = torch.zeros((B, H, W), dtype=torch.bool, device=images.device)
                out = self.model(self._NT(images, mask))
                return out["pred_boxes"], out["pred_logits"]

        return _DetWrapper()

    else:  # panoptic
        class _PanWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self._NT = NestedTensor

            def forward(self, images: torch.Tensor):
                B, _, H, W = images.shape
                mask = torch.zeros((B, H, W), dtype=torch.bool, device=images.device)
                out = self.model(self._NT(images, mask))
                return out["pred_boxes"], out["pred_logits"], out["pred_masks"]

        return _PanWrapper()


# ─────────────────────────────────────────────
# Model catalogue helpers
# ─────────────────────────────────────────────

def print_model_table() -> None:
    """Print a formatted table of all available models."""
    col = 28
    header = (
        f"  {'Variant':<{col}}  {'Task':<10}  {'Backbone':<16}  "
        f"{'Shape':<10}  {'Params(M)':<10}  {'AP50:95':<8}  {'AP50/PQ':<8}  "
        f"{'Lat(ms)':<9}  {'License'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("  Available DETR model variants")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, info in MODEL_CATALOG.items():
        h, w = info["shape"]
        lat  = f"{info['latency_ms']:.0f}" if info["latency_ms"] else "—"
        ap50 = f"{info['ap50']:.1f}" if info["ap50"] is not None else f"PQ {info['pq']:.1f}"
        print(
            f"  {key:<{col}}  {info['task']:<10}  {info['backbone']:<16}  "
            f"{h}×{w:<5}  {info['params_m']:<10.1f}  {info['ap50_95']:<8.1f}  "
            f"{ap50:<8}  {lat:<9}  {info['license']}"
        )
    print("=" * len(header) + "\n")
    print("  Latency measured on V100 GPU with TorchScript transformer.")
    print("  DC5 = dilated conv in last ResNet block (higher-res features, slower).")
    print("  AP values for detection on COCO val2017; PQ for panoptic on COCO val2017.\n")


# ─────────────────────────────────────────────
# Core export
# ─────────────────────────────────────────────

def export_model(
    model_key: str,
    output_dir: str,
    shape: tuple[int, int] | None,
    opset: int,
    batch_size: int,
    verbose: bool,
    custom_weights: str | None,
    force: bool,
    force_hub_reload: bool,
) -> str:
    """Load a DETR model via torch.hub and export it to ONNX.

    Pretrained COCO weights are downloaded automatically by torch.hub unless
    *custom_weights* is provided.

    Args:
        model_key       : Key from MODEL_CATALOG (e.g. "detr_resnet50").
        output_dir      : Final destination directory for the .onnx file.
        shape           : Custom (height, width) or None to use model default.
        opset           : ONNX opset version.
        batch_size      : Batch size embedded in the exported graph.
        verbose         : Show torch.hub download/loading messages.
        custom_weights  : Path to a local .pth checkpoint; None = COCO pretrained.
        force           : Re-export even if the destination .onnx already exists.
        force_hub_reload: Force re-download of the DETR repo via torch.hub.

    Returns:
        Absolute path of the saved .onnx file.
    """
    import torch

    info      = MODEL_CATALOG[model_key]
    hub_name  = info["hub_name"]
    task      = info["task"]

    # ── Resolve export shape ──────────────────────────────────────────────────
    export_shape = shape if shape is not None else info["shape"]
    h, w = export_shape

    # ── Build destination path ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    shape_tag = f"_{h}x{w}" if shape is not None else ""
    dst_name  = f"{model_key}{shape_tag}.onnx"
    dst_path  = os.path.join(output_dir, dst_name)

    if not force and os.path.exists(dst_path):
        print(f"[SKIP] {dst_name} already exists. Use --force to re-export.\n")
        return dst_path

    print(f"[INFO] Model variant  : {model_key}")
    print(f"[INFO] Backbone       : {info['backbone']}")
    print(f"[INFO] Task           : {task}")
    print(f"[INFO] Input shape    : {h}×{w}  (batch {batch_size})")
    print(f"[INFO] ONNX opset     : {opset}")
    if custom_weights:
        print(f"[INFO] Weights        : {custom_weights}")
    else:
        print(f"[INFO] Weights        : COCO pretrained (auto-downloaded)")
        print(f"[INFO] Weight URL     : {info['pth_url']}")
    print()

    # ── Load model via torch.hub ──────────────────────────────────────────────
    print("[INFO] Loading model via torch.hub …")
    print("[INFO] (First run will clone the DETR repo and download ~160–240 MB weights)")
    if not verbose:
        import warnings
        warnings.filterwarnings("ignore")

    load_kwargs: dict = {
        "pretrained": custom_weights is None,
        "force_reload": force_hub_reload,
    }
    try:
        model = torch.hub.load(
            _HUB_REPO, hub_name, trust_repo=True, verbose=verbose, **load_kwargs
        )
    except TypeError:
        # PyTorch < 1.12 does not have trust_repo / verbose kwargs
        model = torch.hub.load(_HUB_REPO, hub_name, **load_kwargs)

    if custom_weights:
        print(f"[INFO] Loading custom weights from: {custom_weights}")
        checkpoint = torch.load(custom_weights, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)

    # Disable aux_loss to keep ONNX output clean (no aux_outputs in graph)
    model.aux_loss = False
    if hasattr(model, "detr"):
        model.detr.aux_loss = False

    model.eval()
    print("[INFO] Model ready.\n")

    # ── Import NestedTensor from DETR source ──────────────────────────────────
    detr_root = _add_detr_to_path()
    if verbose:
        print(f"[INFO] DETR source    : {detr_root}")
    try:
        from util.misc import NestedTensor  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"[ERROR] Could not import NestedTensor from DETR source.\n"
            f"        Expected util/misc.py inside: {detr_root}\n"
            f"        Error: {exc}"
        )
        sys.exit(1)

    # ── Build ONNX wrapper ────────────────────────────────────────────────────
    wrapper = _make_wrapper(model, NestedTensor, task)
    wrapper.eval()

    # ── Dummy input ───────────────────────────────────────────────────────────
    dummy = torch.zeros(batch_size, 3, h, w)

    output_names = (
        ["pred_boxes", "pred_logits", "pred_masks"]
        if task == "panoptic"
        else ["pred_boxes", "pred_logits"]
    )

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"[INFO] Exporting to ONNX (opset {opset}) …")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            dst_path,
            input_names  = ["images"],
            output_names = output_names,
            opset_version = opset,
            do_constant_folding = True,
        )

    # ── Optional ONNX validation ──────────────────────────────────────────────
    try:
        import onnx  # noqa: PLC0415
        onnx_model = onnx.load(dst_path)
        onnx.checker.check_model(onnx_model)
        print("[INFO] ONNX model validation passed.")
    except ImportError:
        pass  # onnx not available; skip validation
    except Exception as exc:
        print(f"[WARN] ONNX validation: {exc}")

    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"\n[SUCCESS] ONNX model saved to : {dst_path}  ({size_mb:.1f} MB)\n")
    return dst_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    default_output = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=(
            "Export DETR pretrained ONNX models.\n\n"
            "Models are loaded via torch.hub (requires internet on first use).\n"
            "Pretrained COCO weights are downloaded automatically from\n"
            "dl.fbaipublicfiles.com.  Run --list-models to see all variants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --model detr_resnet50\n"
            "  %(prog)s --model detr_resnet50 detr_resnet101\n"
            "  %(prog)s --model detr_resnet50_dc5 detr_resnet101_dc5\n"
            "  %(prog)s --model detr_resnet50_panoptic detr_resnet101_panoptic\n"
            "  %(prog)s --model detr_resnet50 --shape 800 1333\n"
            "  %(prog)s --model detr_resnet50 --weights /path/to/checkpoint.pth\n"
            "  %(prog)s --model detr_resnet50 --opset 18 --output-dir ./exports\n"
            "  %(prog)s --list-models"
        ),
    )

    # ── Model selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        nargs="+",
        default=[DEFAULT_MODEL],
        choices=list(MODEL_CATALOG.keys()),
        metavar="VARIANT",
        help=(
            f"Model variant(s) to export. Default: {DEFAULT_MODEL}. "
            "Run --list-models to see all options."
        ),
    )

    # ── Export parameters ─────────────────────────────────────────────────────
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help=(
            "Custom input resolution (height width). "
            "DETR is flexible with input sizes; 800×800 gives best accuracy. "
            "Default: each model's native 800×800."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        metavar="N",
        help="ONNX opset version. Default: 17.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size embedded in the exported ONNX graph. Default: 1.",
    )

    # ── Weight source ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help=(
            "Path to a local .pth checkpoint (format: {'model': state_dict, ...}). "
            "When omitted the official COCO pretrained weights are downloaded "
            "automatically from dl.fbaipublicfiles.com via torch.hub."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        default=default_output,
        metavar="DIR",
        help=f"Directory where .onnx files will be saved. Default: {default_output}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-export even if the destination .onnx file already exists.",
    )

    # ── Hub options ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--force-hub-reload",
        action="store_true",
        default=False,
        help=(
            "Force torch.hub to re-clone the DETR repository and re-download "
            "weights, bypassing the local cache. Use if the cache is corrupted."
        ),
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress torch.hub download messages.",
    )

    # ── Utility ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="Print the model catalogue table and exit.",
    )

    return parser


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_models:
        print_model_table()
        return

    # ── Warn when --weights is used with multiple models ─────────────────────
    if args.weights and len(args.model) > 1:
        print(
            "[WARN] --weights applies the same checkpoint to every model in "
            "--model.\n       This is unusual; pass a single --model variant "
            "when using custom weights."
        )

    # ── Install dependencies ──────────────────────────────────────────────────
    ensure_dependencies()

    # ── Export each model ─────────────────────────────────────────────────────
    shape      = (args.shape[0], args.shape[1]) if args.shape else None
    output_dir = os.path.abspath(args.output_dir)

    exported: list[str] = []
    failed:   list[str] = []

    for model_key in args.model:
        if '_dc5' in model_key:
            print(f"[WARN] Model {model_key} is a DC5 variant and is temporarily disabled because TIDL does not support it. Skipping.")
            continue
    
        print(f"\n{'='*60}")
        print(f"  Exporting: {model_key}")
        print(f"{'='*60}\n")
    
        try:
            out_path = export_model(
                model_key       = model_key,
                output_dir      = output_dir,
                shape           = shape,
                opset           = args.opset,
                batch_size      = args.batch_size,
                verbose         = not args.quiet,
                custom_weights  = args.weights,
                force           = args.force,
                force_hub_reload = args.force_hub_reload,
            )
            exported.append(out_path)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[ERROR] Export failed for '{model_key}': {exc}")
            failed.append(model_key)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Export Summary")
    print("=" * 60)
    for path in exported:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✔  {os.path.basename(path)}  ({size_mb:.1f} MB)")
        print(f"       {path}")
    if failed:
        for key in failed:
            print(f"  ✘  {key}  (FAILED)")
    print("=" * 60 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
