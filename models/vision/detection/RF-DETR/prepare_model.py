"""Script to export RF-DETR pretrained ONNX model(s).

The rfdetr package auto-downloads pretrained COCO weights from HuggingFace on
first use, then this script calls model.export() to produce the ONNX file.

Detection variants (Apache 2.0, COCO pretrained):
  rfdetr_nano    – 384×384,  30.5 M params,  AP50:95 48.4
  rfdetr_small   – 512×512,  32.1 M params,  AP50:95 53.0
  rfdetr_medium  – 576×576,  33.7 M params,  AP50:95 54.7
  rfdetr_large   – 704×704,  33.9 M params,  AP50:95 56.5
  rfdetr_xlarge  – 700×700, 126.4 M params,  AP50:95 58.6  [requires --plus]
  rfdetr_2xlarge – 880×880, 126.9 M params,  AP50:95 60.1  [requires --plus]

Segmentation variants (COCO pretrained):
  rfdetr_seg_nano    – 312×312,  33.6 M params,  AP50:95 40.3
  rfdetr_seg_small   – 384×384,  33.7 M params,  AP50:95 43.1
  rfdetr_seg_medium  – 432×432,  35.7 M params,  AP50:95 45.3
  rfdetr_seg_large   – 504×504,  36.2 M params,  AP50:95 47.1
  rfdetr_seg_xlarge  – 624×624,  38.1 M params,  AP50:95 48.8  [requires --plus]
  rfdetr_seg_2xlarge – 768×768,  38.6 M params,  AP50:95 49.9  [requires --plus]

Input shape constraints:
  Each model's spatial dimensions must be divisible by its block_size
  (= patch_size × num_windows). The rfdetr API enforces this; using a
  custom --shape that violates the constraint will raise a clear error.

Usage:
  python prepare_model.py
  python prepare_model.py --model rfdetr_nano
  python prepare_model.py --model rfdetr_nano rfdetr_small rfdetr_medium
  python prepare_model.py --model rfdetr_large --shape 640 640
  python prepare_model.py --model rfdetr_nano --backbone-only
  python prepare_model.py --model rfdetr_nano --opset 18 --output-dir ./exports
  python prepare_model.py --model rfdetr_xlarge rfdetr_2xlarge --plus
  python prepare_model.py --model rfdetr_seg_nano rfdetr_seg_small
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

# Each entry:  variant_key → (class_name, default_shape, params_m, coco_ap5095, license, requires_plus, task)
MODEL_CATALOG: dict[str, dict] = {
    # ── Detection ────────────────────────────────────────────────────────────
    "rfdetr_nano": {
        "class":        "RFDETRNano",
        "shape":        (384, 384),
        "params_m":     30.5,
        "ap50_95":      48.4,
        "ap50":         67.6,
        "latency_ms":   2.3,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "detection",
    },
    "rfdetr_small": {
        "class":        "RFDETRSmall",
        "shape":        (512, 512),
        "params_m":     32.1,
        "ap50_95":      53.0,
        "ap50":         72.1,
        "latency_ms":   3.5,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "detection",
    },
    "rfdetr_medium": {
        "class":        "RFDETRMedium",
        "shape":        (576, 576),
        "params_m":     33.7,
        "ap50_95":      54.7,
        "ap50":         73.6,
        "latency_ms":   4.4,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "detection",
    },
    "rfdetr_large": {
        "class":        "RFDETRLarge",
        "shape":        (704, 704),
        "params_m":     33.9,
        "ap50_95":      56.5,
        "ap50":         75.1,
        "latency_ms":   6.8,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "detection",
    },
    "rfdetr_xlarge": {
        "class":        "RFDETRXLarge",
        "shape":        (700, 700),
        "params_m":     126.4,
        "ap50_95":      58.6,
        "ap50":         77.4,
        "latency_ms":   11.5,
        "license":      "PML 1.0",
        "requires_plus": True,
        "task":         "detection",
    },
    "rfdetr_2xlarge": {
        "class":        "RFDETR2XLarge",
        "shape":        (880, 880),
        "params_m":     126.9,
        "ap50_95":      60.1,
        "ap50":         78.5,
        "latency_ms":   17.2,
        "license":      "PML 1.0",
        "requires_plus": True,
        "task":         "detection",
    },
    # ── Segmentation ─────────────────────────────────────────────────────────
    "rfdetr_seg_nano": {
        "class":        "RFDETRSegNano",
        "shape":        (312, 312),
        "params_m":     33.6,
        "ap50_95":      40.3,
        "ap50":         63.0,
        "latency_ms":   3.4,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "segmentation",
    },
    "rfdetr_seg_small": {
        "class":        "RFDETRSegSmall",
        "shape":        (384, 384),
        "params_m":     33.7,
        "ap50_95":      43.1,
        "ap50":         66.2,
        "latency_ms":   4.4,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "segmentation",
    },
    "rfdetr_seg_medium": {
        "class":        "RFDETRSegMedium",
        "shape":        (432, 432),
        "params_m":     35.7,
        "ap50_95":      45.3,
        "ap50":         68.4,
        "latency_ms":   5.9,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "segmentation",
    },
    "rfdetr_seg_large": {
        "class":        "RFDETRSegLarge",
        "shape":        (504, 504),
        "params_m":     36.2,
        "ap50_95":      47.1,
        "ap50":         70.5,
        "latency_ms":   8.8,
        "license":      "Apache 2.0",
        "requires_plus": False,
        "task":         "segmentation",
    },
    "rfdetr_seg_xlarge": {
        "class":        "RFDETRSegXLarge",
        "shape":        (624, 624),
        "params_m":     38.1,
        "ap50_95":      48.8,
        "ap50":         72.2,
        "latency_ms":   13.5,
        "license":      "PML 1.0",
        "requires_plus": True,
        "task":         "segmentation",
    },
    "rfdetr_seg_2xlarge": {
        "class":        "RFDETRSeg2XLarge",
        "shape":        (768, 768),
        "params_m":     38.6,
        "ap50_95":      49.9,
        "ap50":         73.1,
        "latency_ms":   21.8,
        "license":      "PML 1.0",
        "requires_plus": True,
        "task":         "segmentation",
    },
}

# Default model exported when --model is not supplied
DEFAULT_MODEL = "rfdetr_nano"


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


def ensure_rfdetr(need_plus: bool = False) -> None:
    """
    Ensure the rfdetr package (with ONNX export support) is importable.
    Installs automatically if missing.

    Args:
        need_plus: Also install the rfdetr[plus] extra required by XLarge /
                   2XLarge models (PML 1.0 licence).
    """
    extras = "rfdetr[onnx,plus]" if need_plus else "rfdetr[onnx]"

    try:
        importlib.import_module("rfdetr")
        print(f"[DEP]  ✔  rfdetr is already installed.")
    except ImportError:
        print(f"[DEP]  ✘  rfdetr not found – installing '{extras}' …")
        _pip_install(extras)
        return

    # rfdetr is present; check the onnx export extra (rfdetr.export module)
    try:
        importlib.import_module("rfdetr.export")
        print("[DEP]  ✔  rfdetr export module is available.")
    except (ImportError, ModuleNotFoundError):
        print(f"[DEP]  ✘  rfdetr export support missing – upgrading to '{extras}' …")
        _pip_install(extras)
        # Invalidate cached module so export_model picks up the upgraded version
        sys.modules.pop("rfdetr", None)
        sys.modules.pop("rfdetr.export", None)


# ─────────────────────────────────────────────
# Model catalogue helpers
# ─────────────────────────────────────────────

def print_model_table() -> None:
    """Print a formatted table of all available models."""
    col = 22
    header = (
        f"  {'Variant':<{col}}  {'Task':<12}  {'Shape':<10}  "
        f"{'Params(M)':<10}  {'AP50:95':<8}  {'AP50':<6}  "
        f"{'Lat(ms)':<8}  {'License'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("  Available RF-DETR model variants")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, info in MODEL_CATALOG.items():
        h, w = info["shape"]
        plus = "  [--plus]" if info["requires_plus"] else ""
        print(
            f"  {key:<{col}}  {info['task']:<12}  {h}×{w:<5}  "
            f"{info['params_m']:<10.1f}  {info['ap50_95']:<8.1f}  "
            f"{info['ap50']:<6.1f}  {info['latency_ms']:<8.1f}  "
            f"{info['license']}{plus}"
        )
    print("=" * len(header) + "\n")
    print("  Latency measured on NVIDIA T4 GPU (TensorRT FP16).")
    print("  [--plus] models require: pip install rfdetr[onnx,plus]\n")


# ─────────────────────────────────────────────
# Core export
# ─────────────────────────────────────────────

def export_model(
    model_key: str,
    output_dir: str,
    shape: tuple[int, int] | None,
    opset: int,
    backbone_only: bool,
    batch_size: int,
    verbose: bool,
    custom_weights: str | None,
    force: bool,
) -> str:
    """
    Instantiate an RF-DETR model and export it to ONNX.

    Pretrained COCO weights are downloaded automatically from HuggingFace
    on first use unless *custom_weights* is provided.

    The rfdetr API always writes the ONNX to:
        <tmp_dir>/inference_model.onnx  (or backbone_model.onnx)

    This function moves it to:
        <output_dir>/<model_key>.onnx   (or <model_key>_backbone.onnx)

    Any .pth weight files produced by rfdetr in the temp directory are also
    kept and moved to output_dir alongside the ONNX.

    Args:
        model_key      : Key from MODEL_CATALOG (e.g. "rfdetr_nano").
        output_dir     : Final destination directory for the .onnx file.
        shape          : Custom (height, width) or None to use model default.
        opset          : ONNX opset version.
        backbone_only  : Export backbone feature extractor only.
        batch_size     : Batch size embedded in the exported graph.
        verbose        : Print rfdetr's internal export messages.
        custom_weights : Path to a local .pth checkpoint; None = COCO pretrained.
        force          : Re-export even if the destination .onnx already exists.

    Returns:
        Absolute path of the saved .onnx file.
    """
    import rfdetr  # noqa: PLC0415

    info       = MODEL_CATALOG[model_key]
    class_name = info["class"]
    task       = info["task"]

    # ── Resolve the model class ───────────────────────────────────────────────
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        print(
            f"[ERROR] Class '{class_name}' not found in the rfdetr package.\n"
            f"        Make sure rfdetr is up-to-date: pip install -U rfdetr[onnx]"
        )
        sys.exit(1)

    # ── Determine export shape ────────────────────────────────────────────────
    export_shape = shape if shape is not None else info["shape"]
    h, w = export_shape

    # ── Build destination path early so we can check for an existing file ─────
    os.makedirs(output_dir, exist_ok=True)
    suffix   = "_backbone" if backbone_only else ""
    shape_tag = f"_{h}x{w}" if shape is not None else ""
    dst_name = f"{model_key}{shape_tag}{suffix}.onnx"
    dst_path = os.path.join(output_dir, dst_name)

    if not force and os.path.exists(dst_path):
        print(f"[SKIP] {dst_name} already exists. Use --force to re-export.\n")
        return dst_path

    print(f"[INFO] Model class : {class_name}")
    print(f"[INFO] Task        : {task}")
    print(f"[INFO] Input shape : {h}×{w}")
    print(f"[INFO] Opset       : {opset}")
    print(f"[INFO] Batch size  : {batch_size}")
    if backbone_only:
        print("[INFO] Mode        : backbone only")
    if custom_weights:
        print(f"[INFO] Weights     : {custom_weights}")
    else:
        print("[INFO] Weights     : COCO pretrained (auto-downloaded)")
    print()

    # ── Instantiate model ─────────────────────────────────────────────────────
    print("[INFO] Loading model …")
    init_kwargs: dict = {}
    if custom_weights:
        init_kwargs["pretrain_weights"] = custom_weights
    model = model_cls(**init_kwargs)
    print("[INFO] Model ready.\n")

    # ── Export to a temporary directory, then move to final destination ───────
    with tempfile.TemporaryDirectory(prefix="rfdetr_export_") as tmp_dir:
        print("[INFO] Exporting to ONNX …")
        model.export(
            output_dir     = tmp_dir,
            format         = "onnx",
            shape          = (h, w),
            opset_version  = opset,
            backbone_only  = backbone_only,
            batch_size     = batch_size,
            verbose        = verbose,
        )

        # Locate the exported .onnx file.
        # rfdetr has used different output filenames across versions:
        #   ≤1.x  → inference_model.onnx / backbone_model.onnx
        #   ≥1.8  → rfdetr-<size>.onnx  (e.g. rfdetr-nano.onnx)
        # Try known names first, then fall back to any .onnx in the directory.
        size_tag = model_key.replace("rfdetr_seg_", "").replace("rfdetr_", "")
        candidates_ordered = (
            [f"rfdetr-{size_tag}.onnx", "backbone_model.onnx"]
            if backbone_only else
            [f"rfdetr-{size_tag}.onnx", "inference_model.onnx"]
        )

        src_path = None
        for name in candidates_ordered:
            p = os.path.join(tmp_dir, name)
            if os.path.exists(p):
                src_path = p
                break

        if src_path is None:
            all_onnx = [f for f in os.listdir(tmp_dir) if f.endswith(".onnx")]
            if not all_onnx:
                print(f"[ERROR] No .onnx file found in temp dir: {tmp_dir}")
                sys.exit(1)
            src_path = os.path.join(tmp_dir, all_onnx[0])
            print(f"[INFO] Located exported model: {all_onnx[0]}")

        shutil.move(src_path, dst_path)

        # Keep any .pth weight files produced alongside the ONNX
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".pth"):
                pth_dst = os.path.join(output_dir, fname)
                shutil.move(os.path.join(tmp_dir, fname), pth_dst)
                print(f"[INFO] PTH weights saved to : {pth_dst}")

    print(f"\n[SUCCESS] ONNX model saved to: {dst_path}\n")
    return dst_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    default_output = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=(
            "Export RF-DETR pretrained ONNX models.\n\n"
            "Pretrained COCO weights are downloaded automatically from\n"
            "HuggingFace on first use.  Run --list-models to see all variants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --model rfdetr_nano\n"
            "  %(prog)s --model rfdetr_nano rfdetr_small rfdetr_large\n"
            "  %(prog)s --model rfdetr_medium --shape 608 608\n"
            "  %(prog)s --model rfdetr_nano --backbone-only\n"
            "  %(prog)s --model rfdetr_xlarge rfdetr_2xlarge --plus\n"
            "  %(prog)s --model rfdetr_seg_nano rfdetr_seg_medium\n"
            "  %(prog)s --model rfdetr_large --weights /path/to/custom.pth\n"
            "  %(prog)s --model rfdetr_nano --opset 18 --output-dir ./exports\n"
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
            "Custom input resolution (height width). Must be divisible by the "
            "model's block_size (patch_size × num_windows). "
            "Default: each model's native resolution."
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
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        default=False,
        help=(
            "Export only the DINOv2 backbone (feature extractor). "
            "Output is named <variant>_backbone.onnx."
        ),
    )

    # ── Weight source ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help=(
            "Path to a local .pth checkpoint. "
            "When omitted the official COCO pretrained weights are downloaded "
            "automatically from HuggingFace."
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

    # ── Plus models ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--plus",
        action="store_true",
        default=False,
        help=(
            "Install rfdetr[onnx,plus] to enable XLarge / 2XLarge models "
            "(PML 1.0 license). Required when exporting rfdetr_xlarge, "
            "rfdetr_2xlarge, rfdetr_seg_xlarge, or rfdetr_seg_2xlarge."
        ),
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress rfdetr's internal verbose output.",
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

    # ── Validate requested models ─────────────────────────────────────────────
    need_plus = args.plus or any(
        MODEL_CATALOG[m]["requires_plus"] for m in args.model
    )

    plus_models = [m for m in args.model if MODEL_CATALOG[m]["requires_plus"]]
    if plus_models and not args.plus:
        print(
            f"[WARN] {', '.join(plus_models)} require the rfdetr[plus] extra "
            "(PML 1.0 license).\n"
            "       Re-run with --plus to confirm and install it."
        )
        sys.exit(1)

    # ── Warn when --weights is used with multiple models ─────────────────────
    if args.weights and len(args.model) > 1:
        print(
            "[WARN] --weights applies the same checkpoint to every model in "
            "--model.\n       This is unusual; pass a single --model variant "
            "when using custom weights."
        )

    # ── Install rfdetr ────────────────────────────────────────────────────────
    ensure_rfdetr(need_plus=need_plus)

    # ── Export each model ─────────────────────────────────────────────────────
    shape = (args.shape[0], args.shape[1]) if args.shape else None
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
                backbone_only  = args.backbone_only,
                batch_size     = args.batch_size,
                verbose        = not args.quiet,
                custom_weights = args.weights,
                force          = args.force,
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
