"""Script to export DEIMv2 pretrained ONNX model(s).

DEIMv2 (Real-Time Object Detection Meets DINOv3) from Intellindust AI Lab.
Models are loaded via HuggingFace Hub using the PyTorchModelHubMixin interface.

The script:
  1. Clones the DEIMv2 source repo to ~/.cache/deimv2_src (once).
  2. Downloads pretrained COCO weights from HuggingFace on first use.
  3. Exports each variant to ONNX with two named outputs:
       pred_boxes  [1, num_queries, 4]  – CxCyWH normalised [0,1]
       pred_logits [1, num_queries, 80] – raw class logits

Model variants (Apache 2.0, COCO pretrained):
  deimv2_atto   – 320×320,  0.5M params,  AP50:95 23.8
  deimv2_femto  – 416×416,  1.0M params,  AP50:95 31.0
  deimv2_pico   – 640×640,  1.5M params,  AP50:95 38.5
  deimv2_n      – 640×640,  3.6M params,  AP50:95 43.0
  deimv2_s      – 640×640,  9.7M params,  AP50:95 50.9  [default]
  deimv2_m      – 640×640, 18.1M params,  AP50:95 53.0
  deimv2_l      – 640×640, 32.2M params,  AP50:95 56.0
  deimv2_x      – 640×640, 50.3M params,  AP50:95 57.8

Usage:
  python prepare_model.py
  python prepare_model.py --model deimv2_s
  python prepare_model.py --model deimv2_s deimv2_m deimv2_l
  python prepare_model.py --model deimv2_l --shape 800 800
  python prepare_model.py --model deimv2_x --opset 18 --output-dir ./exports
  python prepare_model.py --model deimv2_s --weights /path/to/checkpoint.pth
  python prepare_model.py --model all
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
# Source repo configuration
# ─────────────────────────────────────────────

_DEIMV2_REPO_URL  = "https://github.com/Intellindust-AI-Lab/DEIMv2.git"
_DEIMV2_CACHE_DIR = os.path.expanduser("~/.cache/deimv2_src")


# ─────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────

# Each entry: variant_key → metadata dict
MODEL_CATALOG: dict[str, dict] = {
    "deimv2_atto": {
        "hf_repo":     "Intellindust/DEIMv2_HGNetv2_ATTO_COCO",
        "backbone":    "HGNetv2-Atto",
        "shape":       (320, 320),
        "params_m":    0.5,
        "flops_g":     0.8,
        "ap50_95":     23.8,
        "latency_ms":  1.10,
        "num_queries": 100,
        "license":     "Apache 2.0",
    },
    "deimv2_femto": {
        "hf_repo":     "Intellindust/DEIMv2_HGNetv2_FEMTO_COCO",
        "backbone":    "HGNetv2-Femto",
        "shape":       (416, 416),
        "params_m":    1.0,
        "flops_g":     1.7,
        "ap50_95":     31.0,
        "latency_ms":  1.45,
        "num_queries": 150,
        "license":     "Apache 2.0",
    },
    "deimv2_pico": {
        "hf_repo":     "Intellindust/DEIMv2_HGNetv2_PICO_COCO",
        "backbone":    "HGNetv2-Pico",
        "shape":       (640, 640),
        "params_m":    1.5,
        "flops_g":     5.2,
        "ap50_95":     38.5,
        "latency_ms":  2.13,
        "num_queries": 200,
        "license":     "Apache 2.0",
    },
    "deimv2_n": {
        "hf_repo":     "Intellindust/DEIMv2_HGNetv2_N_COCO",
        "backbone":    "HGNetv2-B0",
        "shape":       (640, 640),
        "params_m":    3.6,
        "flops_g":     6.8,
        "ap50_95":     43.0,
        "latency_ms":  2.32,
        "num_queries": 300,
        "license":     "Apache 2.0",
    },
    "deimv2_s": {
        "hf_repo":     "Intellindust/DEIMv2_DINOv3_S_COCO",
        "backbone":    "DINOv3-vit_tiny",
        "shape":       (640, 640),
        "params_m":    9.7,
        "flops_g":     25.6,
        "ap50_95":     50.9,
        "latency_ms":  5.78,
        "num_queries": 300,
        "license":     "Apache 2.0",
    },
    "deimv2_m": {
        "hf_repo":     "Intellindust/DEIMv2_DINOv3_M_COCO",
        "backbone":    "DINOv3-vit_tinyplus",
        "shape":       (640, 640),
        "params_m":    18.1,
        "flops_g":     52.2,
        "ap50_95":     53.0,
        "latency_ms":  8.80,
        "num_queries": 300,
        "license":     "Apache 2.0",
    },
    "deimv2_l": {
        "hf_repo":     "Intellindust/DEIMv2_DINOv3_L_COCO",
        "backbone":    "DINOv3-vit_small",
        "shape":       (640, 640),
        "params_m":    32.2,
        "flops_g":     96.7,
        "ap50_95":     56.0,
        "latency_ms":  10.47,
        "num_queries": 300,
        "license":     "Apache 2.0",
    },
    "deimv2_x": {
        "hf_repo":     "Intellindust/DEIMv2_DINOv3_X_COCO",
        "backbone":    "DINOv3-vit_small+",
        "shape":       (640, 640),
        "params_m":    50.3,
        "flops_g":     151.6,
        "ap50_95":     57.8,
        "latency_ms":  13.75,
        "num_queries": 300,
        "license":     "Apache 2.0",
    },
}

DEFAULT_MODEL = "deimv2_s"


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
        "onnx":             "onnx",
        "huggingface_hub":  "huggingface_hub",
        "timm":             "timm",       # needed for DINOv3 ViT backbone
        "calflops":         "calflops",   # required by DEIMv2 engine modules
        "onnxslim":         "onnxslim",   # preferred simplifier for ViT-based models
        "onnxsim":          "onnxsim",    # fallback simplifier for HGNetv2-based models
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
# DEIMv2 source management
# ─────────────────────────────────────────────

def ensure_deimv2_source() -> str:
    """
    Clone (or reuse) the DEIMv2 repository to *_DEIMV2_CACHE_DIR*.

    Returns the cache directory path that should be prepended to sys.path
    before importing engine.backbone, engine.deim, etc.

    The repo is cloned once and reused across invocations.  If the
    target directory already exists it is left as-is to keep the
    environment reproducible.
    """
    if os.path.isdir(_DEIMV2_CACHE_DIR):
        print(f"[SOURCE]  ✔  DEIMv2 source found at: {_DEIMV2_CACHE_DIR}\n")
        return _DEIMV2_CACHE_DIR

    print(f"[SOURCE] Cloning DEIMv2 repository to: {_DEIMV2_CACHE_DIR} …")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", _DEIMV2_REPO_URL, _DEIMV2_CACHE_DIR],
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("[SOURCE] ERROR: git clone failed.")
        if result.stderr:
            print(result.stderr.strip())
        print("[SOURCE] Ensure git is installed and the network is reachable.")
        print(f"         URL: {_DEIMV2_REPO_URL}")
        sys.exit(1)

    print(f"[SOURCE] Repository cloned.\n")
    return _DEIMV2_CACHE_DIR


def inject_source_path(src_dir: str) -> None:
    """Prepend *src_dir* to sys.path so `from engine.backbone import ...` works."""
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# ─────────────────────────────────────────────
# Model catalogue helpers
# ─────────────────────────────────────────────

def print_model_table() -> None:
    """Print a formatted table of all available models."""
    col = 14
    header = (
        f"  {'Variant':<{col}}  {'Backbone':<20}  {'Shape':<10}  "
        f"{'Params(M)':<10}  {'FLOPs(G)':<9}  {'AP50:95':<8}  {'Lat(ms)'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("  Available DEIMv2 model variants")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, info in MODEL_CATALOG.items():
        h, w = info["shape"]
        print(
            f"  {key:<{col}}  {info['backbone']:<20}  {h}×{w:<5}  "
            f"{info['params_m']:<10.1f}  {info['flops_g']:<9.1f}  "
            f"{info['ap50_95']:<8.1f}  {info['latency_ms']:.2f}"
        )
    print("=" * len(header) + "\n")
    print("  AP evaluated on COCO val2017.")
    print("  Latency measured on NVIDIA A100 GPU (TensorRT FP16, batch=1).\n")


# ─────────────────────────────────────────────
# DEIMv2 class builder
# ─────────────────────────────────────────────

def _build_deimv2_class():
    """Import engine modules and define DEIMv2 class with HuggingFace mixin.

    Returns the DEIMv2 class (not an instance).
    """
    import torch.nn as nn
    from huggingface_hub import PyTorchModelHubMixin

    try:
        from engine.backbone import HGNetv2, DINOv3STAs
        from engine.deim import HybridEncoder, LiteEncoder
        from engine.deim import DFINETransformer, DEIMTransformer
        from engine.deim.postprocessor import PostProcessor
    except ImportError as exc:
        print(f"[ERROR] Could not import from DEIMv2 source: {exc}")
        print(f"[ERROR] Ensure DEIMv2 source is at: {_DEIMV2_CACHE_DIR}")
        sys.exit(1)

    class DEIMv2(nn.Module, PyTorchModelHubMixin):
        """DEIMv2 model with HuggingFace Hub integration.

        Architecture:
          backbone → encoder → decoder → postprocessor

        For export we skip the postprocessor to get raw decoder outputs:
          pred_boxes  [B, num_queries, 4]  CxCyWH normalised [0,1]
          pred_logits [B, num_queries, 80] raw class logits
        """
        def __init__(self, config):
            super().__init__()
            # Backbone
            if 'DINOv3STAs' in config:
                self.backbone = DINOv3STAs(**config["DINOv3STAs"])
            else:
                self.backbone = HGNetv2(**config["HGNetv2"])
            # Encoder
            if 'LiteEncoder' in config:
                self.encoder = LiteEncoder(**config["LiteEncoder"])
            else:
                self.encoder = HybridEncoder(**config["HybridEncoder"])
            # Decoder
            if 'DEIMTransformer' in config:
                self.decoder = DEIMTransformer(**config["DEIMTransformer"])
            else:
                self.decoder = DFINETransformer(**config["DFINETransformer"])
            # PostProcessor (not used in ONNX export)
            self.postprocessor = PostProcessor(**config["PostProcessor"])

        def forward(self, x, orig_target_sizes):
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.postprocessor(x, orig_target_sizes)
            return x

    return DEIMv2


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
    """Simplify the ONNX model using onnxslim (preferred) or onnxsim (fallback).

    onnxsim fails on DINOv3 ViT backbones (S/M/L/X variants) with:
      "IndexError: Input /backbone/rope_embed/Sub_1_output_0 is undefined!"
    Root cause: the DINOv3 ViT backbone computes RoPE (Rotary Position
    Embedding) with a Sub node whose input is a dynamic slice of the positional
    grid.  onnxsim's constant-folding pass evaluates this symbolically and
    silently drops the intermediate tensor, leaving a dangling reference.
    onnxslim uses a different graph optimization strategy that preserves
    dynamic nodes, so it handles ViT models correctly.  HGNetv2-based models
    (Atto/Femto/Pico/N) don't use RoPE so onnxsim works fine on those.
    """
    # ── Try onnxslim first (handles RoPE / dynamic nodes in ViT backbones) ────
    try:
        import onnxslim
        print("[POST] Simplifying ONNX model with onnxslim …")
        onnxslim.slim(onnx_path, onnx_path)
        print("[POST] Simplification complete (onnxslim).\n")
        return
    except ImportError:
        print("[POST] onnxslim not installed – trying onnxsim …")
    except Exception as exc:
        print(f"[POST] WARNING: onnxslim failed ({exc}) – trying onnxsim …")

    # ── Fall back to onnxsim ──────────────────────────────────────────────────
    try:
        import onnx
        import onnxsim
    except ImportError:
        print("[POST] onnxsim not installed either – skipping simplification.\n")
        print("[POST] Install a simplifier:  pip install onnxslim  (recommended)\n")
        return

    print("[POST] Simplifying ONNX model with onnxsim …")
    try:
        model = onnx.load(onnx_path)
        model_simp, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_simp, onnx_path)
            print("[POST] Simplification complete (onnxsim).\n")
        else:
            print("[POST] WARNING: onnxsim validation failed – using original.\n")
    except Exception as exc:
        print(f"[POST] WARNING: onnxsim failed ({exc}) – using original.\n")
        print("[POST] For ViT-based models (S/M/L/X), use: pip install onnxslim\n")


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
    Load DEIMv2 model from HuggingFace and export to ONNX.

    The exported graph has a single image input and two outputs:
      pred_boxes  [B, num_queries, 4]  – CxCyWH normalised [0,1]
      pred_logits [B, num_queries, 80] – raw class logits

    Shape inference always runs. onnxsim simplification runs by default.

    Args:
        model_key     : Key from MODEL_CATALOG (e.g. "deimv2_s").
        output_dir    : Directory where the .onnx file will be saved.
        shape         : Custom (H, W) override, or None for model default.
        opset         : ONNX opset version (default 17).
        batch_size    : Batch size in the exported graph (default 1).
        verbose       : Print detailed loading messages.
        custom_weights: Path to a local .pth checkpoint; None = HF pretrained.
        force         : Re-export even if the destination .onnx already exists.
        simplify      : Apply onnxsim after export (default: True).

    Returns:
        Absolute path of the saved .onnx file.
    """
    import torch
    import torch.nn as nn

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
    print(f"[INFO] HF repo       : {info['hf_repo']}")
    print(f"[INFO] Input shape   : {export_h}×{export_w}")
    print(f"[INFO] Batch size    : {batch_size}")
    print(f"[INFO] Num queries   : {info['num_queries']}")
    print(f"[INFO] ONNX opset    : {opset}")
    print()

    # ── Ensure DEIMv2 source is available ─────────────────────────────────────
    src_dir = ensure_deimv2_source()
    inject_source_path(src_dir)

    # ── Build DEIMv2 class ────────────────────────────────────────────────────
    DEIMv2 = _build_deimv2_class()

    # ── Load model from HuggingFace ───────────────────────────────────────────
    print("[INFO] Loading model from HuggingFace Hub …")
    print("[INFO] (First run will download ~10–200 MB weights)")

    if custom_weights:
        # Load HF config + override weights from local file
        print(f"[INFO] Loading architecture from HF, weights from: {custom_weights}")
        model = DEIMv2.from_pretrained(info["hf_repo"])
        checkpoint = torch.load(custom_weights, map_location="cpu", weights_only=False)
        state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        if isinstance(state, dict) and "module" in state:
            state = state["module"]
        model.load_state_dict(state)
    else:
        model = DEIMv2.from_pretrained(info["hf_repo"])

    print("[INFO] Model loaded.\n")

    # ── Build ONNX export wrapper ─────────────────────────────────────────────
    # Skip postprocessor to get raw decoder outputs (pred_boxes, pred_logits)
    class _OnnxWrapper(nn.Module):
        """ONNX wrapper: backbone + encoder + decoder only (no postprocessor)."""
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.backbone = m.backbone
            self.encoder  = m.encoder
            self.decoder  = m.decoder

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.backbone(images)
            x = self.encoder(x)
            out = self.decoder(x)
            # DEIMTransformer / DFINETransformer at eval returns dict:
            #   {"pred_boxes": [B, N, 4], "pred_logits": [B, N, 80]}
            return out["pred_boxes"], out["pred_logits"]

    model.eval()
    wrapper = _OnnxWrapper(model).eval()

    # ── Dry-run to confirm output shapes ──────────────────────────────────────
    dummy = torch.zeros(batch_size, 3, export_h, export_w)
    with torch.no_grad():
        boxes_out, logits_out = wrapper(dummy)
    print(f"[INFO] pred_boxes  shape : {list(boxes_out.shape)}")
    print(f"[INFO] pred_logits shape : {list(logits_out.shape)}")
    print()

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print(f"[INFO] Exporting to ONNX (opset {opset}) …")
    with tempfile.TemporaryDirectory(prefix="deimv2_export_") as tmp_dir:
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
            "Export DEIMv2 pretrained ONNX models.\n\n"
            "Pretrained COCO weights are downloaded automatically from HuggingFace\n"
            "Hub on first use.  Run --list-models to see all variants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --model deimv2_s\n"
            "  %(prog)s --model deimv2_s deimv2_m deimv2_l\n"
            "  %(prog)s --model deimv2_l --shape 800 800\n"
            "  %(prog)s --model deimv2_x --opset 18 --output-dir ./exports\n"
            "  %(prog)s --model deimv2_s --weights /path/to/custom.pth\n"
            "  %(prog)s --model all\n"
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
            "Default: each model's native resolution (320×320, 416×416, or 640×640)."
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
            "When omitted the official COCO pretrained weights are downloaded "
            "automatically from HuggingFace Hub."
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
            "Apply onnx-simplifier after export (best-effort, default: enabled). "
            "Requires: pip install onnxsim. Use --no-simplify to disable."
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

    # ── Expand "all" keyword ─────────────────────────────────────────────────
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
            print(f"  ✘  {key}  (FAILED)")
    print("=" * 60 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
