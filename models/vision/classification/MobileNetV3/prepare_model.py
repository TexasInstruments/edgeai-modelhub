#!/usr/bin/env python3
"""
Export MobileNetV3 classification models from torchvision to ONNX
for TI EdgeAI hardware deployment.

Model variants (BSD-3-Clause, ImageNet-1K pretrained):
  mobilenetv3_large  – 224×224,  5.48M params, 0.22G FLOPs,  top-1  75.274%  [default]

Usage:
  python prepare_model.py
  python prepare_model.py --model mobilenetv3_large
  python prepare_model.py --model mobilenetv3_large --shape 224 224
  python prepare_model.py --model all
  python prepare_model.py --model mobilenetv3_large --weights /path/to/custom.pth
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


# ─────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────

MODEL_CATALOG: dict[str, dict] = {
    "mobilenetv3_large": {
        "tv_weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        "backbone":   "MobileNetV3-Large",
        "shape":      (224, 224),
        "params_m":   5.48,
        "flops_g":    0.22,
        "top1_acc":   75.274,
        "top5_acc":   92.566,
        "license":    "BSD-3-Clause",
    },
    # "mobilenetv3_small": excluded — produces poor accuracy under TIDL compilation
    # {
    #     "tv_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    #     "backbone":   "MobileNetV3-Small",
    #     "shape":      (224, 224),
    #     "params_m":   2.54,
    #     "flops_g":    0.06,
    #     "top1_acc":   67.668,
    #     "top5_acc":   87.402,
    #     "license":    "BSD-3-Clause",
    # },
}

DEFAULT_MODEL = "mobilenetv3_large"


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
        "torch":       "torch",
        "torchvision": "torchvision",
        "onnx":        "onnx",
        "onnxsim":     "onnx-simplifier",
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
    """Simplify the ONNX model in-place using onnxsim."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        print("[POST] onnxsim not installed – skipping simplification.\n")
        print("[POST] Install with: pip install onnx-simplifier\n")
        return

    print("[POST] Simplifying ONNX model with onnxsim …")
    try:
        model = onnx.load(onnx_path)
        model_simp, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_simp, onnx_path)
            print("[POST] Simplification complete.\n")
        else:
            print("[POST] WARNING: onnxsim validation failed – using original.\n")
    except Exception as exc:
        print(f"[POST] WARNING: onnxsim failed ({exc}) – using original.\n")


# ─────────────────────────────────────────────
# Model catalogue helpers
# ─────────────────────────────────────────────

def print_model_table() -> None:
    """Print a formatted table of all available models."""
    col = 22
    header = (
        f"  {'Variant':<{col}}  {'Backbone':<18}  {'Shape':<10}  "
        f"{'Params(M)':<10}  {'FLOPs(G)':<9}  {'Top-1 %':<9}  Top-5 %"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("  Available MobileNetV3 model variants")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, info in MODEL_CATALOG.items():
        h, w = info["shape"]
        print(
            f"  {key:<{col}}  {info['backbone']:<18}  {h}×{w:<5}  "
            f"{info['params_m']:<10.2f}  {info['flops_g']:<9.2f}  "
            f"{info['top1_acc']:<9.3f}  {info['top5_acc']:.3f}"
        )
    print("=" * len(header) + "\n")
    print("  Accuracy evaluated on ImageNet-1K val (torchvision pretrained weights).")
    print("  License: BSD-3-Clause (torchvision / PyTorch).\n")


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
    simplify: bool = True,
) -> str:
    """
    Load MobileNetV3 model from torchvision and export to ONNX.

    The exported graph has a single image input (NCHW) and one output:
      output  [batch_size, 1000]  – raw class logits (ImageNet-1K)

    Shape inference and optional onnxsim simplification are applied.

    Args:
        model_key     : Key from MODEL_CATALOG (e.g. "mobilenetv3_large").
        output_dir    : Directory where the .onnx file will be saved.
        shape         : Custom (H, W) override, or None for model default.
        opset         : ONNX opset version (default 17).
        batch_size    : Batch size in the exported graph (default 1).
        verbose       : Print detailed loading messages.
        custom_weights: Path to a local .pth checkpoint; None = torchvision pretrained.
        force         : Re-export even if the destination .onnx already exists.
        simplify      : Apply onnxsim after export (default: True).

    Returns:
        Absolute path of the saved .onnx file.
    """
    import torch
    import torchvision.models as tvm

    info = MODEL_CATALOG[model_key]
    export_h, export_w = shape if shape is not None else info["shape"]

    # ── Destination path ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    shape_tag = f"_{export_h}x{export_w}" if shape is not None else ""
    dst_name  = f"{model_key}{shape_tag}.onnx"
    dst_path  = os.path.join(output_dir, dst_name)

    if not force and os.path.exists(dst_path):
        print(f"[SKIP] {dst_name} already exists. Use --force to re-export.\n")
        return dst_path

    print(f"[INFO] Model variant : {model_key}")
    print(f"[INFO] Backbone      : {info['backbone']}")
    print(f"[INFO] TV weights    : {info['tv_weights']}")
    print(f"[INFO] Input shape   : {export_h}×{export_w}")
    print(f"[INFO] Batch size    : {batch_size}")
    print(f"[INFO] ONNX opset    : {opset}")
    print()

    # ── Load model ───────────────────────────────────────────────────────────
    if custom_weights:
        print(f"[INFO] Loading architecture from torchvision, weights from: {custom_weights}")
        if model_key == "mobilenetv3_large":
            model = tvm.mobilenet_v3_large(weights=None)
        else:
            model = tvm.mobilenet_v3_small(weights=None)
        checkpoint = torch.load(custom_weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        if isinstance(state, dict) and "module" in state:
            state = state["module"]
        model.load_state_dict(state)
    else:
        print(f"[INFO] Loading pretrained weights from torchvision …")
        print(f"[INFO] (First run may download weights ~10–20 MB)")
        if model_key == "mobilenetv3_large":
            weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            model   = tvm.mobilenet_v3_large(weights=weights)
        else:
            weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model   = tvm.mobilenet_v3_small(weights=weights)

    model.eval()
    print(f"[INFO] Model loaded.\n")

    # ── Dry-run to confirm output shape ──────────────────────────────────────
    dummy = torch.zeros(batch_size, 3, export_h, export_w)
    with torch.no_grad():
        out = model(dummy)
    print(f"[INFO] Output shape : {list(out.shape)}")
    print()

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print(f"[INFO] Exporting to ONNX (opset {opset}) …")
    with tempfile.TemporaryDirectory(prefix="mobilenetv3_export_") as tmp_dir:
        tmp_path = os.path.join(tmp_dir, dst_name)

        torch.onnx.export(
            model,
            dummy,
            tmp_path,
            input_names=["input"],
            output_names=["output"],
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
            "Export MobileNetV3 pretrained ONNX models.\n\n"
            "Pretrained ImageNet-1K weights are downloaded automatically from\n"
            "torchvision on first use.  Run --list-models to see all variants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --model mobilenetv3_large\n"
            "  %(prog)s --model mobilenetv3_large mobilenetv3_small\n"
            "  %(prog)s --model mobilenetv3_large --shape 224 224\n"
            "  %(prog)s --model all\n"
            "  %(prog)s --model mobilenetv3_large --weights /path/to/custom.pth\n"
            "  %(prog)s --list-models"
        ),
    )

    # ── Model selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        nargs="+",
        default=[DEFAULT_MODEL],
        choices=list(MODEL_CATALOG.keys()) + ["all"],
        metavar="VARIANT",
        help=(
            f"Model variant(s) to export. Use 'all' for all variants. "
            f"Default: {DEFAULT_MODEL}. Run --list-models to see all options."
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
            "Default: each model's native resolution (224×224)."
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
            "Path to a local .pth checkpoint (optional). "
            "When omitted the official ImageNet-1K pretrained weights are "
            "downloaded automatically from torchvision."
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
        default=True,
        help=(
            "Apply onnx-simplifier after export (default: enabled). "
            "Requires: pip install onnx-simplifier. Use --no-simplify to disable."
        ),
    )
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="Disable onnx-simplifier after export.",
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

    # ── Expand "all" keyword ──────────────────────────────────────────────────
    if "all" in args.model:
        args.model = list(MODEL_CATALOG.keys())

    # ── Warn when --weights is used with multiple models ──────────────────────
    if args.weights and len(args.model) > 1:
        print(
            "[WARN] --weights applies the same checkpoint to every model in "
            "--model.\n       This is unusual; pass a single --model variant "
            "when using custom weights.\n"
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
            print(f"  ✘  {key}")
    print(f"\n  {len(exported)}/{len(args.model)} model(s) exported successfully.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
