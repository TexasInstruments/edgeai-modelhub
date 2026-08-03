"""Script to export RT-DETRv2 pretrained ONNX model(s).

RT-DETRv2 is a real-time object detection transformer from:
  "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time
   Detection Transformer" (arXiv:2407.17140, CVPR 2024)

This script:
  1. Clones the official RT-DETR source from GitHub (cached in ~/.cache/rtdetr_src).
  2. Downloads pretrained COCO weights from GitHub Releases.
  3. Exports each variant to ONNX with two named outputs:
       pred_boxes  [1, 300, 4]  – CxCyWH normalised [0,1]
       pred_logits [1, 300, 80] – raw class logits

Model variants (Apache 2.0, COCO pretrained):
  rtdetrv2_s   – 640×640,  20 M params,  AP50:95 48.1  [default]
  rtdetrv2_ms  – 640×640,  31 M params,  AP50:95 49.9  (M* lighter variant)
  rtdetrv2_m   – 640×640,  36 M params,  AP50:95 51.9
  rtdetrv2_l   – 640×640,  42 M params,  AP50:95 53.4
  rtdetrv2_x   – 640×640,  76 M params,  AP50:95 54.3

FPS measured on NVIDIA T4, TensorRT FP16.

Usage:
  python prepare_model.py
  python prepare_model.py --model rtdetrv2_s
  python prepare_model.py --model rtdetrv2_s rtdetrv2_m rtdetrv2_l
  python prepare_model.py --model rtdetrv2_l --shape 800 800
  python prepare_model.py --model rtdetrv2_s --opset 18 --output-dir ./exports
  python prepare_model.py --model rtdetrv2_x --weights /path/to/custom.pth
  python prepare_model.py --list-models
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request


# ─────────────────────────────────────────────
# Source repo configuration
# ─────────────────────────────────────────────

_RTDETR_REPO_URL  = "https://github.com/lyuwenyu/RT-DETR.git"
_RTDETR_CACHE_DIR = os.path.expanduser("~/.cache/rtdetr_src")

# Config files live at:  <_RTDETR_CACHE_DIR>/rtdetrv2_pytorch/configs/rtdetrv2/
_CONFIG_SUBDIR     = os.path.join("rtdetrv2_pytorch", "configs", "rtdetrv2")
# Source code lives at: <_RTDETR_CACHE_DIR>/rtdetrv2_pytorch/
_SRC_SUBDIR        = "rtdetrv2_pytorch"

# Pretrained weight download bases
_BASE_V02 = "https://github.com/lyuwenyu/storage/releases/download/v0.2"
_BASE_V01 = "https://github.com/lyuwenyu/storage/releases/download/v0.1"


# ─────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────

# Each entry: variant_key → metadata dict
MODEL_CATALOG: dict[str, dict] = {
    "rtdetrv2_s": {
        "backbone":    "ResNet-18vd",
        "config":      "rtdetrv2_r18vd_120e_coco.yml",
        "weight_url":  f"{_BASE_V02}/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "weight_file": "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "shape":       (640, 640),
        "params_m":    20.0,
        "flops_g":     60.0,
        "ap50_95":     48.1,
        "ap50":        65.1,
        "fps_t4":      217,
        "num_queries": 300,
    },
    "rtdetrv2_ms": {
        "backbone":    "ResNet-34vd",
        "config":      "rtdetrv2_r34vd_120e_coco.yml",
        "weight_url":  f"{_BASE_V01}/rtdetrv2_r34vd_120e_coco_ema.pth",
        "weight_file": "rtdetrv2_r34vd_120e_coco_ema.pth",
        "shape":       (640, 640),
        "params_m":    31.0,
        "flops_g":     92.0,
        "ap50_95":     49.9,
        "ap50":        67.5,
        "fps_t4":      161,
        "num_queries": 300,
    },
    "rtdetrv2_m": {
        "backbone":    "ResNet-50vd-m",
        "config":      "rtdetrv2_r50vd_m_7x_coco.yml",
        "weight_url":  f"{_BASE_V01}/rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "weight_file": "rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "shape":       (640, 640),
        "params_m":    36.0,
        "flops_g":     100.0,
        "ap50_95":     51.9,
        "ap50":        69.9,
        "fps_t4":      145,
        "num_queries": 300,
    },
    "rtdetrv2_l": {
        "backbone":    "ResNet-50vd",
        "config":      "rtdetrv2_r50vd_6x_coco.yml",
        "weight_url":  f"{_BASE_V01}/rtdetrv2_r50vd_6x_coco_ema.pth",
        "weight_file": "rtdetrv2_r50vd_6x_coco_ema.pth",
        "shape":       (640, 640),
        "params_m":    42.0,
        "flops_g":     136.0,
        "ap50_95":     53.4,
        "ap50":        71.6,
        "fps_t4":      108,
        "num_queries": 300,
    },
    "rtdetrv2_x": {
        "backbone":    "ResNet-101vd",
        "config":      "rtdetrv2_r101vd_6x_coco.yml",
        "weight_url":  f"{_BASE_V01}/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
        "weight_file": "rtdetrv2_r101vd_6x_coco_from_paddle.pth",
        "shape":       (640, 640),
        "params_m":    76.0,
        "flops_g":     259.0,
        "ap50_95":     54.3,
        "ap50":        72.8,
        "fps_t4":      74,
        "num_queries": 300,
    },
}

DEFAULT_MODEL = "rtdetrv2_s"


# ─────────────────────────────────────────────
# Dependency installer
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
    """Ensure all runtime dependencies are available."""
    needed: list[str] = []
    checks = {
        "torch":            "torch",
        "scipy":            "scipy",
        "yaml":             "PyYAML",
        "onnx":             "onnx",
        "faster_coco_eval": "faster-coco-eval",
    }
    for mod, pkg in checks.items():
        try:
            importlib.import_module(mod)
            print(f"[DEP]  ✔  {mod} is already installed.")
        except ImportError:
            print(f"[DEP]  ✘  {mod} not found – will install '{pkg}'.")
            needed.append(pkg)
    if needed:
        _pip_install(*needed)
    else:
        print("[DEP] All dependencies satisfied.\n")


# ─────────────────────────────────────────────
# RT-DETR source management
# ─────────────────────────────────────────────

def ensure_rtdetr_source() -> str:
    """
    Clone (or update) the RT-DETR repository to *_RTDETR_CACHE_DIR*.

    Returns the path to the rtdetrv2_pytorch sub-directory that must be
    prepended to sys.path before importing src.core.

    The repo is cloned once and reused across invocations.  If the
    target directory already exists it is left as-is (no auto-pull) to
    keep the environment reproducible.
    """
    src_dir = os.path.join(_RTDETR_CACHE_DIR, _SRC_SUBDIR)

    if os.path.isdir(src_dir):
        print(f"[SOURCE]  ✔  RT-DETR source found at: {src_dir}\n")
        return src_dir

    print(f"[SOURCE] Cloning RT-DETR repository to: {_RTDETR_CACHE_DIR} …")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", _RTDETR_REPO_URL, _RTDETR_CACHE_DIR],
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("[SOURCE] ERROR: git clone failed.")
        if result.stderr:
            print(result.stderr.strip())
        print("[SOURCE] Ensure git is installed and the network is reachable.")
        print(f"         URL: {_RTDETR_REPO_URL}")
        sys.exit(1)

    print(f"[SOURCE] Repository cloned.\n")
    return src_dir


def inject_source_path(src_dir: str) -> None:
    """Prepend *src_dir* to sys.path so `from src.core import YAMLConfig` works."""
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# ─────────────────────────────────────────────
# Weight downloader
# ─────────────────────────────────────────────

def _download_weights(url: str, dest: str) -> None:
    """Download a checkpoint from *url* to *dest* with a progress indicator."""
    if os.path.exists(dest):
        print(f"[WEIGHTS]  ✔  Checkpoint already at: {dest}")
        return

    print(f"[WEIGHTS] Downloading pretrained weights …")
    print(f"          URL : {url}")
    print(f"          Dest: {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    try:
        def _progress(count: int, block: int, total: int) -> None:
            if total > 0:
                pct = min(100, count * block * 100 // total)
                print(f"\r[WEIGHTS] {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r[WEIGHTS] 100%  →  saved to: {dest}\n")
    except Exception as exc:
        if os.path.exists(dest):
            os.remove(dest)
        print(f"\n[WEIGHTS] ERROR: download failed: {exc}")
        print(f"[WEIGHTS] Download manually from: {url}")
        print(f"[WEIGHTS] and place it at:        {dest}")
        sys.exit(1)


# ─────────────────────────────────────────────
# Model catalogue helpers
# ─────────────────────────────────────────────

def print_model_table() -> None:
    """Print a formatted table of all available models."""
    col = 14
    header = (
        f"  {'Variant':<{col}}  {'Backbone':<14}  {'Shape':<10}  "
        f"{'Params(M)':<10}  {'FLOPs(G)':<9}  {'AP50:95':<8}  "
        f"{'AP50':<6}  {'FPS(T4)'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("  Available RT-DETRv2 model variants")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, info in MODEL_CATALOG.items():
        h, w = info["shape"]
        print(
            f"  {key:<{col}}  {info['backbone']:<14}  {h}×{w:<5}  "
            f"{info['params_m']:<10.0f}  {info['flops_g']:<9.0f}  "
            f"{info['ap50_95']:<8.1f}  {info['ap50']:<6.1f}  "
            f"{info['fps_t4']}"
        )
    print("=" * len(header) + "\n")
    print("  AP evaluated on COCO val2017.")
    print("  FPS measured on NVIDIA T4 GPU (TensorRT FP16, batch=1).\n")


# ─────────────────────────────────────────────
# ONNX post-processing helpers
# ─────────────────────────────────────────────

def _run_shape_inference(onnx_path: str) -> None:
    """Run ONNX shape inference in-place."""
    try:
        import onnx
        import onnx.shape_inference
        print("[POST] Running ONNX shape inference …")
        model = onnx.load(onnx_path)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, onnx_path)
        print("[POST] Shape inference complete.\n")
    except Exception as exc:
        print(f"[POST] WARNING: shape inference failed ({exc}) – model unchanged.\n")


def _maybe_simplify(onnx_path: str) -> None:
    """Optionally simplify the ONNX model using onnxsim (best-effort)."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        print("[POST] onnxsim not installed – skipping simplification.\n")
        return

    print("[POST] Simplifying ONNX model …")
    try:
        model = onnx.load(onnx_path)
        model_simp, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_simp, onnx_path)
            print("[POST] Simplification complete.\n")
        else:
            print("[POST] WARNING: simplification validation failed – using original.\n")
    except Exception as exc:
        print(f"[POST] WARNING: simplification failed ({exc}) – using original.\n")


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
    simplify: bool,
) -> str:
    """
    Download weights (if needed), load the RT-DETRv2 model, and export to ONNX.

    The exported graph has a single image input and two outputs:
      pred_boxes  [B, num_queries, 4]  – CxCyWH normalised [0,1]
      pred_logits [B, num_queries, 80] – raw class logits

    Args:
        model_key     : Key from MODEL_CATALOG (e.g. "rtdetrv2_l").
        output_dir    : Directory where the .onnx file will be saved.
        shape         : Custom (H, W) override, or None for model default.
        opset         : ONNX opset version (default 16).
        batch_size    : Batch size in the exported graph (default 1).
        verbose       : Print RT-DETR's internal loading messages.
        custom_weights: Path to a local .pth checkpoint; None = official COCO weights.
        force         : Re-export even if the destination .onnx already exists.
        simplify      : Apply onnxsim after export (best-effort).

    Returns:
        Absolute path of the saved .onnx file.
    """
    import torch
    import torch.nn as nn

    info        = MODEL_CATALOG[model_key]
    export_h, export_w = shape if shape is not None else info["shape"]

    # ── Destination path ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    shape_tag = f"_{export_h}x{export_w}" if shape is not None else ""
    dst_name  = f"{model_key}{shape_tag}.onnx"
    dst_path  = os.path.join(output_dir, dst_name)

    if not force and os.path.exists(dst_path):
        print(f"[SKIP] {dst_name} already exists. Use --force to re-export.\n")
        return dst_path

    # ── Resolve weights path ──────────────────────────────────────────────────
    if custom_weights:
        weights_path = custom_weights
        print(f"[INFO] Using custom weights: {weights_path}")
    else:
        weights_path = os.path.join(output_dir, info["weight_file"])
        _download_weights(info["weight_url"], weights_path)

    # ── Ensure RT-DETR source is available ────────────────────────────────────
    src_dir = ensure_rtdetr_source()
    inject_source_path(src_dir)

    # ── Import model infrastructure ───────────────────────────────────────────
    print("[INFO] Loading RT-DETRv2 model infrastructure …")
    try:
        from src.core import YAMLConfig  # noqa: PLC0415
    except ImportError as exc:
        print(f"[ERROR] Could not import from RT-DETR source: {exc}")
        print(f"        Source directory: {src_dir}")
        sys.exit(1)

    # ── Build config ──────────────────────────────────────────────────────────
    config_path = os.path.join(_RTDETR_CACHE_DIR, _CONFIG_SUBDIR, info["config"])
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        print("[ERROR] The RT-DETR source clone may be incomplete.")
        sys.exit(1)

    if verbose:
        print(f"[INFO] Config     : {config_path}")
        print(f"[INFO] Backbone   : {info['backbone']}")
        print(f"[INFO] Input shape: {export_h}×{export_w}")
        print(f"[INFO] Opset      : {opset}")
        print(f"[INFO] Batch size : {batch_size}")
        print(f"[INFO] Queries    : {info['num_queries']}")
        print()

    cfg = YAMLConfig(config_path)

    # ── Load pretrained weights ───────────────────────────────────────────────
    print("[INFO] Loading weights …")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    cfg.model.load_state_dict(state)
    print("[INFO] Weights loaded.\n")

    # ── Build deploy-mode export wrapper ──────────────────────────────────────
    # cfg.model.deploy() removes training-only components (EMA, label assignment).
    # We return (pred_boxes, pred_logits) as separate outputs so downstream
    # YAML configs can apply sigmoid and box decoding independently.
    class _ExportWrapper(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.model(images)
            # pred_boxes  : [B, num_queries, 4]  CxCyWH normalised [0,1]
            # pred_logits : [B, num_queries, num_classes]
            return out["pred_boxes"], out["pred_logits"]

    deploy_model = cfg.model.deploy()
    wrapper = _ExportWrapper(deploy_model)
    wrapper.eval()

    # ── Dry-run to confirm output shapes ─────────────────────────────────────
    dummy = torch.zeros(batch_size, 3, export_h, export_w)
    with torch.no_grad():
        boxes_out, logits_out = wrapper(dummy)
    print(f"[INFO] pred_boxes  shape : {list(boxes_out.shape)}")
    print(f"[INFO] pred_logits shape : {list(logits_out.shape)}")
    print()

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print(f"[INFO] Exporting to ONNX (opset {opset}) …")
    with tempfile.TemporaryDirectory(prefix="rtdetrv2_export_") as tmp_dir:
        tmp_path = os.path.join(tmp_dir, dst_name)

        torch.onnx.export(
            wrapper,
            dummy,
            tmp_path,
            input_names=["images"],
            output_names=["pred_boxes", "pred_logits"],
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )

        shutil.move(tmp_path, dst_path)

    print(f"[INFO] Raw ONNX written to: {dst_path}")

    # ── Post-processing ───────────────────────────────────────────────────────
    _run_shape_inference(dst_path)
    if simplify:
        _maybe_simplify(dst_path)

    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"\n[SUCCESS] ONNX model saved to: {dst_path}  ({size_mb:.1f} MB)\n")
    return dst_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    default_output = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=(
            "Export RT-DETRv2 pretrained ONNX models.\n\n"
            "Pretrained COCO weights are downloaded automatically from GitHub\n"
            "Releases on first use.  Run --list-models to see all variants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --model rtdetrv2_s\n"
            "  %(prog)s --model rtdetrv2_s rtdetrv2_m rtdetrv2_l\n"
            "  %(prog)s --model rtdetrv2_l --shape 800 800\n"
            "  %(prog)s --model rtdetrv2_s --opset 18 --output-dir ./exports\n"
            "  %(prog)s --model rtdetrv2_x --weights /path/to/custom.pth\n"
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
            "Default: 640×640 for all variants."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        metavar="N",
        help="ONNX opset version. Default: 16.",
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
            "Path to a local .pth checkpoint. "
            "When omitted the official COCO pretrained weights are downloaded "
            "automatically from GitHub Releases."
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

    # ── Simplification ────────────────────────────────────────────────────────
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
        help=(
            "Apply onnx-simplifier after export (best-effort). "
            "Requires: pip install onnxsim"
        ),
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress verbose output during model loading.",
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
        print(f"\n{'='*60}")
        print(f"  Exporting: {model_key}")
        print(f"{'='*60}\n")

        try:
            out_path = export_model(
                model_key      = model_key,
                output_dir     = output_dir,
                shape          = shape,
                opset          = args.opset,
                batch_size     = args.batch_size,
                verbose        = not args.quiet,
                custom_weights = args.weights,
                force          = args.force,
                simplify       = args.simplify,
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
