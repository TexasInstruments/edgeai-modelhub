"""Script to export Deformable-DETR pretrained ONNX model(s).

Deformable DETR (Deformable Transformers for End-to-End Object Detection)
from SenseTime / fundamentalvision.

Reference: https://github.com/fundamentalvision/Deformable-DETR
Paper: https://arxiv.org/abs/2010.04159

Detection variants (Apache 2.0, COCO pretrained):
  deformable_detr_single_scale                          – 800×800, 34M params, AP50:95 39.4
  deformable_detr_single_scale_dc5                      – 800×800, 34M params, AP50:95 41.5
  deformable_detr                                       – 800×800, 40M params, AP50:95 44.5
  deformable_detr_plus_iterative_bbox_refinement        – 800×800, 41M params, AP50:95 46.2
  deformable_detr_two_stage                             – 800×800, 41M params, AP50:95 46.9

Export methods (--method):
  torch   (default) – clones the official GitHub repo, downloads weights from
                      Google Drive via gdown.  Requires internet access to
                      drive.google.com (may be blocked on corporate proxies).
  optimum           – downloads from HuggingFace Hub via the transformers
                      library.  Proxy-friendly, no CUDA compilation, no Google
                      Drive.  Uses HuggingFace model IDs under SenseTime/.

Notes:
  - All variants use a ResNet-50 backbone, pre-trained on ImageNet.
  - DC5 variant is disabled: TIDL does not support dilated convolution in ResNet.

Usage:
  python prepare_model.py
  python prepare_model.py --method optimum
  python prepare_model.py --model deformable_detr_single_scale
  python prepare_model.py --model deformable_detr_single_scale --method optimum
  python prepare_model.py --model deformable_detr deformable_detr_two_stage
  python prepare_model.py --model deformable_detr --shape 640 640
  python prepare_model.py --model deformable_detr --weights /path/to/checkpoint.pth
  python prepare_model.py --model deformable_detr --opset 18 --output-dir ./exports
  python prepare_model.py --model deformable_detr --skip-simplify
  python prepare_model.py --list-models
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import subprocess
import sys


# ─────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────

# Each entry: variant_key → metadata dict
#   num_feature_levels : 1 (single scale) or 4 (multi-scale)
#   with_box_refine    : iterative bounding box refinement
#   two_stage          : two-stage proposal + detection
#   dilation           : DC5 – dilation in ResNet's last block
#   gdrive_id          : Google Drive file ID (used by --method torch)
#   hf_model_id        : HuggingFace model ID (used by --method optimum)
MODEL_CATALOG: dict[str, dict] = {
    "deformable_detr_single_scale": {
        "num_feature_levels": 1,
        "with_box_refine":    False,
        "two_stage":          False,
        "dilation":           False,
        "shape":              (800, 800),
        "params_m":           34,
        "ap50_95":            39.4,
        "flops_g":            78,
        "fps_v100":           27.0,
        "license":            "Apache 2.0",
        "gdrive_id":          "1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU",
        "hf_model_id":        "SenseTime/deformable-detr-single-scale",
    },
    "deformable_detr_single_scale_dc5": {
        "num_feature_levels": 1,
        "with_box_refine":    False,
        "two_stage":          False,
        "dilation":           True,
        "shape":              (800, 800),
        "params_m":           34,
        "ap50_95":            41.5,
        "flops_g":            128,
        "fps_v100":           22.1,
        "license":            "Apache 2.0",
        "gdrive_id":          "1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk",
        "hf_model_id":        "SenseTime/deformable-detr-single-scale-dc5",
    },
    "deformable_detr": {
        "num_feature_levels": 4,
        "with_box_refine":    False,
        "two_stage":          False,
        "dilation":           False,
        "shape":              (800, 800),
        "params_m":           40,
        "ap50_95":            44.5,
        "flops_g":            173,
        "fps_v100":           15.0,
        "license":            "Apache 2.0",
        "gdrive_id":          "1nDWZWHuRwtwGden77NLM9JoWe-YisJnA",
        "hf_model_id":        "SenseTime/deformable-detr",
    },
    "deformable_detr_plus_iterative_bbox_refinement": {
        "num_feature_levels": 4,
        "with_box_refine":    True,
        "two_stage":          False,
        "dilation":           False,
        "shape":              (800, 800),
        "params_m":           41,
        "ap50_95":            46.2,
        "flops_g":            173,
        "fps_v100":           15.0,
        "license":            "Apache 2.0",
        "gdrive_id":          "1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI",
        "hf_model_id":        "SenseTime/deformable-detr-with-box-refine",
    },
    "deformable_detr_two_stage": {
        "num_feature_levels": 4,
        "with_box_refine":    True,
        "two_stage":          True,
        "dilation":           False,
        "shape":              (800, 800),
        "params_m":           41,
        "ap50_95":            46.9,
        "flops_g":            173,
        "fps_v100":           14.5,
        "license":            "Apache 2.0",
        "gdrive_id":          "15I03A7hNTpwuLNdfuEmW9_taZMNVssEp",
        "hf_model_id":        "SenseTime/deformable-detr-with-box-refine-two-stage",
    },
}

DEFAULT_MODEL = "deformable_detr"

_REPO_URL       = "https://github.com/fundamentalvision/Deformable-DETR.git"
_REPO_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "deformable_detr")


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
    """Ensure torch, torchvision, onnx, scipy, gdown, and onnxsim are importable."""
    required = [
        ("torch",       "torch>=1.12.0"),
        ("torchvision", "torchvision>=0.13.0"),
        ("onnx",        "onnx>=1.14.0"),
        ("onnxsim",     "onnx-simplifier"),
        ("scipy",       "scipy"),
        ("gdown",       "gdown>=5.2.0"),
    ]
    missing = []
    for mod_name, pip_spec in required:
        try:
            importlib.import_module(mod_name)
            print(f"[DEP]  ✔  {mod_name} is installed.")
        except ImportError:
            print(f"[DEP]  ✘  {mod_name} not found.")
            missing.append(pip_spec)

    if missing:
        _pip_install(*missing)
    print()


def ensure_hf_dependencies() -> None:
    """Ensure torch, torchvision, onnx, onnxsim, and transformers are importable
    (needed for --method optimum)."""
    required = [
        ("torch",        "torch>=1.12.0"),
        ("torchvision",  "torchvision>=0.13.0"),
        ("onnx",         "onnx>=1.14.0"),
        ("onnxsim",      "onnx-simplifier"),
        ("transformers", "transformers>=4.30.0"),
    ]
    missing = []
    for mod_name, pip_spec in required:
        try:
            importlib.import_module(mod_name)
            print(f"[DEP]  ✔  {mod_name} is installed.")
        except ImportError:
            print(f"[DEP]  ✘  {mod_name} not found.")
            missing.append(pip_spec)

    if missing:
        _pip_install(*missing)
    print()


# ─────────────────────────────────────────────
# Repo setup
# ─────────────────────────────────────────────

def setup_repo(force_reclone: bool = False) -> str:
    """Clone (or reuse) the Deformable-DETR repository.

    Returns the absolute path to the repository root.
    """
    if os.path.isdir(_REPO_CACHE_DIR) and not force_reclone:
        print(f"[REPO] Using cached repo: {_REPO_CACHE_DIR}")
        return _REPO_CACHE_DIR

    if os.path.isdir(_REPO_CACHE_DIR):
        import shutil
        shutil.rmtree(_REPO_CACHE_DIR)

    os.makedirs(os.path.dirname(_REPO_CACHE_DIR), exist_ok=True)
    print(f"[REPO] Cloning Deformable-DETR into {_REPO_CACHE_DIR} …")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", _REPO_URL, _REPO_CACHE_DIR],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[REPO] ERROR: git clone failed.\n{result.stderr.strip()}")
        sys.exit(1)
    print("[REPO] Clone complete.\n")
    return _REPO_CACHE_DIR


# ─────────────────────────────────────────────
# Torchvision compatibility shim
# ─────────────────────────────────────────────

def _patch_torchvision_compat() -> None:
    """Stub out removed torchvision symbols referenced by Deformable-DETR's
    util/misc.py.

    The repo does  ``float(torchvision.__version__[:3]) < 0.5``  to gate
    old code.  For torchvision >= 0.10  the string ``"0.15"[:3]``  is
    ``"0.1"``  →  float 0.1  →  the condition is True, triggering an import
    of  ``_NewEmptyTensorOp``  that was removed in torchvision 0.9.

    We add a harmless stub so the import succeeds.  The function
    util/misc.interpolate() falls back to ``torch.nn.functional.interpolate``
    for non-empty tensors (all practical cases), so the stub is never called.
    """
    import torch                          # noqa: PLC0415
    import torchvision.ops.misc as _tvm  # noqa: PLC0415

    if hasattr(_tvm, "_NewEmptyTensorOp"):
        return  # already present (old torchvision) — nothing to do

    class _NewEmptyTensorOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, new_size):
            return x.new_empty(new_size)

        @staticmethod
        def backward(ctx, grad):
            return grad, None

    _tvm._NewEmptyTensorOp = _NewEmptyTensorOp
    print("[COMPAT] Added _NewEmptyTensorOp stub to torchvision.ops.misc.\n")


# ─────────────────────────────────────────────
# Python fallback for deformable attention
# ─────────────────────────────────────────────

def _install_python_fallback(repo_root: str) -> None:
    """Set up a pure-Python replacement for the multi-scale deformable
    attention CUDA extension so that ONNX tracing works on CPU without
    requiring CUDA compilation.

    Strategy:
      1. Patch torchvision.ops.misc to add the removed _NewEmptyTensorOp stub
         (required for util/misc.py to import on torchvision >= 0.10).
      2. Register a placeholder 'MultiScaleDeformableAttention' module in
         sys.modules before any model code is imported (models/ops imports
         this at module level).
      3. Import the pure-Python ms_deform_attn_core_pytorch function from
         the repo source.
      4. Monkeypatch MSDeformAttn.forward to call ms_deform_attn_core_pytorch
         directly, bypassing the MSDeformAttnFunction custom autograd op
         (which has no ONNX symbolic and would break torch.onnx.export).
    """
    import types
    import torch                              # noqa: PLC0415

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Step 1 – Fix torchvision compatibility before importing any repo code.
    _patch_torchvision_compat()

    # Step 2 – Register a stub MSDA module so models/ops imports succeed.
    if "MultiScaleDeformableAttention" not in sys.modules:
        stub = types.ModuleType("MultiScaleDeformableAttention")
        sys.modules["MultiScaleDeformableAttention"] = stub
        print("[OPS] Registered stub MultiScaleDeformableAttention module.")

    # Step 3 – Import the Python-only reference implementation.
    from models.ops.functions.ms_deform_attn_func import (  # noqa: PLC0415
        ms_deform_attn_core_pytorch,
    )

    # Step 4 – Patch MSDeformAttn.forward to use ms_deform_attn_core_pytorch
    # directly instead of calling MSDeformAttnFunction.apply.
    # This is a verbatim rewrite of the original forward with only the final
    # output = MSDeformAttnFunction.apply(...) line replaced.
    import torch.nn.functional as F          # noqa: PLC0415
    import models.ops.modules.ms_deform_attn as _attn_mod  # noqa: PLC0415

    def _py_forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, "
                f"got {reference_points.shape[-1]}"
            )

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        return output

    _attn_mod.MSDeformAttn.forward = _py_forward
    print("[OPS] Pure-Python fallback installed for MSDeformAttn (no CUDA required).\n")


# ─────────────────────────────────────────────
# ONNX simplification
# ─────────────────────────────────────────────

def simplify_onnx(src_path: str, force: bool = False) -> bool:
    """Run onnx-simplifier on *src_path* in-place.

    Simplification folds constants, removes dead nodes, and cleans up
    redundant ops produced by torch.onnx.export, making the graph smaller
    and easier to deploy.

    Falls back gracefully (copies as-is) if onnxsim is not installed.

    Args:
        src_path: Path to the .onnx file to simplify (modified in-place).
        force   : Re-run even if the file was already simplified.

    Returns:
        True on success (or if simplification was skipped gracefully).
    """
    print(f"\n[SIM]  Running onnxsim on {os.path.basename(src_path)} …")

    try:
        import onnx      # noqa: PLC0415
        import onnxsim   # noqa: PLC0415
    except ImportError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        print(f"  [WARN] {missing} not installed – skipping simplification.")
        print("         Install with: pip install onnx-simplifier")
        return True

    try:
        model = onnx.load(src_path)
    except Exception as exc:
        print(f"  [ERROR] Failed to load {src_path}: {exc}")
        return False

    try:
        model_sim, check = onnxsim.simplify(model)
    except Exception as exc:
        print(f"  [WARN] onnxsim failed: {exc}  – keeping unsimplified model.")
        return True

    if not check:
        print("  [WARN] onnxsim validation failed – keeping unsimplified model.")
        return True

    orig_nodes = len(model.graph.node)
    sim_nodes  = len(model_sim.graph.node)
    delta      = orig_nodes - sim_nodes
    print(f"  Nodes: {orig_nodes} → {sim_nodes}  (−{delta})")

    try:
        onnx.save(model_sim, src_path)
    except Exception as exc:
        print(f"  [ERROR] Failed to save simplified model: {exc}")
        return False

    size_mb = os.path.getsize(src_path) / (1024 * 1024)
    print(f"[OK]   Simplified model saved: {src_path}  ({size_mb:.1f} MB)")
    return True


# ─────────────────────────────────────────────
# Weight download
# ─────────────────────────────────────────────

def download_weights(gdrive_id: str, weights_path: str, force: bool = False) -> bool:
    """Download a pretrained checkpoint from Google Drive using gdown.

    Args:
        gdrive_id   : Google Drive file ID.
        weights_path: Local destination path for the .pth checkpoint.
        force       : Re-download even if file already exists.

    Returns:
        True on success.
    """
    import gdown  # noqa: PLC0415

    if os.path.exists(weights_path) and not force:
        size_mb = os.path.getsize(weights_path) / 1024 / 1024
        print(f"[SKIP] Weights already exist ({size_mb:.1f} MB). "
              "Use --force to re-download.\n")
        return True

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[DOWN] Downloading pretrained weights from Google Drive …")
    print(f"       File ID : {gdrive_id}")
    print(f"       Dest    : {weights_path}")

    # Pick up proxy settings from the environment (e.g. TI corporate proxy).
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if proxy:
        print(f"       Proxy   : {proxy}")

    try:
        dl_kwargs: dict = {"quiet": False}
        if proxy:
            dl_kwargs["proxy"] = proxy
        gdown.download(url, weights_path, **dl_kwargs)
    except Exception as exc:
        print(f"[ERROR] gdown download failed: {exc}")
        _print_manual_download_hint(gdrive_id, weights_path)
        return False

    if not os.path.exists(weights_path):
        print("[ERROR] Download finished but file was not created.")
        _print_manual_download_hint(gdrive_id, weights_path)
        return False

    size_mb = os.path.getsize(weights_path) / 1024 / 1024
    print(f"[OK]   Weights saved: {weights_path}  ({size_mb:.1f} MB)\n")
    return True


def _print_manual_download_hint(gdrive_id: str, weights_path: str) -> None:
    """Print instructions for manually downloading a Google Drive checkpoint."""
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(
        f"\n[HINT] If you are behind a corporate proxy, download the checkpoint\n"
        f"       manually using one of the following commands:\n"
        f"\n"
        f"         # with gdown and explicit proxy:\n"
        f"         gdown --proxy <proxy_url> '{url}' -O '{weights_path}'\n"
        f"\n"
        f"         # with curl:\n"
        f"         curl -L -x <proxy_url> '{url}' -o '{weights_path}'\n"
        f"\n"
        f"       Then re-run with --weights to skip the download:\n"
        f"         python prepare_model.py --model <variant> --weights '{weights_path}'\n"
    )


# ─────────────────────────────────────────────
# Model catalogue helpers
# ─────────────────────────────────────────────

def print_model_table() -> None:
    """Print a formatted table of all available model variants."""
    col = 48
    header = (
        f"  {'Variant':<{col}}  {'Shape':<10}  {'Params(M)':<10}  "
        f"{'AP50:95':<8}  {'FLOPs(G)':<9}  {'FPS(V100)':<10}  {'License'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("  Available Deformable-DETR model variants")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, info in MODEL_CATALOG.items():
        h, w = info["shape"]
        print(
            f"  {key:<{col}}  {h}×{w:<5}  "
            f"{info['params_m']:<10}  {info['ap50_95']:<8.1f}  "
            f"{info['flops_g']:<9}  {info['fps_v100']:<10.1f}  "
            f"{info['license']}"
        )
    print("=" * len(header) + "\n")
    print("  All variants use ResNet-50 backbone, trained on COCO 2017.")
    print("  AP50:95 measured on COCO val2017, inference speed on V100 GPU.\n")


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
    force_reclone: bool,
    skip_simplify: bool = False,
) -> str:
    """Clone the Deformable-DETR repo, download weights, and export to ONNX.

    Args:
        model_key     : Key from MODEL_CATALOG.
        output_dir    : Directory to save the .onnx and .pth files.
        shape         : Custom (height, width) or None for model default.
        opset         : ONNX opset version.
        batch_size    : Batch size embedded in the exported graph.
        verbose       : Show additional progress messages.
        custom_weights: Path to a local .pth checkpoint; None = pretrained.
        force         : Re-export even if .onnx already exists.
        force_reclone : Force re-clone of the source repo.

    Returns:
        Absolute path of the saved .onnx file.
    """
    import torch  # noqa: PLC0415

    info = MODEL_CATALOG[model_key]
    export_shape = shape if shape is not None else info["shape"]
    h, w = export_shape

    os.makedirs(output_dir, exist_ok=True)
    shape_tag = f"_{h}x{w}" if shape is not None else ""
    dst_name  = f"{model_key}{shape_tag}.onnx"
    dst_path  = os.path.join(output_dir, dst_name)

    if not force and os.path.exists(dst_path):
        print(f"[SKIP] {dst_name} already exists. Use --force to re-export.\n")
        return dst_path

    print(f"[INFO] Variant      : {model_key}")
    print(f"[INFO] Feature lvls : {info['num_feature_levels']}")
    print(f"[INFO] Box refine   : {info['with_box_refine']}")
    print(f"[INFO] Two-stage    : {info['two_stage']}")
    print(f"[INFO] DC5 dilation : {info['dilation']}")
    print(f"[INFO] Input shape  : {h}×{w}  (batch {batch_size})")
    print(f"[INFO] ONNX opset   : {opset}")
    if custom_weights:
        print(f"[INFO] Weights      : {custom_weights}")
    else:
        print(f"[INFO] Weights      : COCO pretrained (Google Drive)")

    # ── Step 1: Clone repo ────────────────────────────────────────────────────
    repo_root = setup_repo(force_reclone=force_reclone)

    # ── Step 2: Install Python fallback for deformable attention ──────────────
    _install_python_fallback(repo_root)

    # ── Step 3: Download or locate weights ────────────────────────────────────
    if custom_weights:
        weights_path = custom_weights
        if not os.path.exists(weights_path):
            print(f"[ERROR] Custom weights not found: {weights_path}")
            sys.exit(1)
    else:
        weights_path = os.path.join(output_dir, f"{model_key}.pth")
        if not download_weights(info["gdrive_id"], weights_path, force=force):
            print(f"[ERROR] Failed to download weights for '{model_key}'.")
            sys.exit(1)

    # ── Step 4: Build model ───────────────────────────────────────────────────
    print("[INFO] Building model …")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from models import build_model  # noqa: PLC0415

    args = argparse.Namespace(
        # Backbone
        backbone                = "resnet50",
        dilation                = info["dilation"],
        position_embedding      = "sine",
        position_embedding_scale= 2 * math.pi,
        num_feature_levels      = info["num_feature_levels"],
        # Transformer
        enc_layers              = 6,
        dec_layers              = 6,
        dim_feedforward         = 1024,
        hidden_dim              = 256,
        dropout                 = 0.1,
        nheads                  = 8,
        num_queries             = 300,
        dec_n_points            = 4,
        enc_n_points            = 4,
        # Variant flags
        with_box_refine         = info["with_box_refine"],
        two_stage               = info["two_stage"],
        # Segmentation (not used for detection export)
        masks                   = False,
        frozen_weights          = None,
        # Loss (needed by SetCriterion constructor, not used for inference)
        aux_loss                = False,
        set_cost_class          = 2.0,
        set_cost_bbox           = 5.0,
        set_cost_giou           = 2.0,
        mask_loss_coef          = 1.0,
        dice_loss_coef          = 1.0,
        cls_loss_coef           = 2.0,
        bbox_loss_coef          = 5.0,
        giou_loss_coef          = 2.0,
        focal_alpha             = 0.25,
        # Dataset (determines num_classes = 91 for coco)
        dataset_file            = "coco",
        coco_path               = "./data/coco",
        coco_panoptic_path      = None,
        remove_difficult        = False,
        # Device
        device                  = "cpu",
    )

    model, _criterion, _postprocessors = build_model(args)
    model.eval()
    print("[INFO] Model built.\n")

    # ── Step 5: Load pretrained weights ───────────────────────────────────────
    print(f"[INFO] Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected if not k.endswith(("total_params", "total_ops"))]
    if missing:
        print(f"[WARN] Missing keys  : {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print("[INFO] Weights loaded.\n")

    # ── Step 6: Build ONNX wrapper ────────────────────────────────────────────
    import torch                                     # noqa: PLC0415
    import torch.nn as nn                            # noqa: PLC0415
    from util.misc import NestedTensor               # noqa: PLC0415

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self._NT  = NestedTensor

        def forward(self, images: torch.Tensor):
            B, _, H, W = images.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=images.device)
            out  = self.model(self._NT(images, mask))
            return out["pred_boxes"], out["pred_logits"]

    wrapper = _Wrapper().eval()

    # ── Step 7: Export to ONNX ────────────────────────────────────────────────
    print(f"[INFO] Exporting to ONNX (opset {opset}) …")
    dummy = torch.zeros(batch_size, 3, h, w)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            dst_path,
            input_names   = ["images"],
            output_names  = ["pred_boxes", "pred_logits"],
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
        pass
    except Exception as exc:
        print(f"[WARN] ONNX validation: {exc}")

    # ── Optional onnxsim simplification ──────────────────────────────────────
    if not skip_simplify:
        simplify_onnx(dst_path, force=force)

    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"\n[SUCCESS] ONNX model saved to: {dst_path}  ({size_mb:.1f} MB)\n")
    return dst_path


# ─────────────────────────────────────────────
# Optimum / HuggingFace export
# ─────────────────────────────────────────────

def export_model_optimum(
    model_key: str,
    output_dir: str,
    shape: tuple[int, int] | None,
    opset: int,
    batch_size: int,
    force: bool,
    skip_simplify: bool = False,
) -> str:
    """Download from HuggingFace and export Deformable-DETR to ONNX.

    Uses the HuggingFace transformers implementation of Deformable DETR,
    which is a pure-Python port of the original architecture.  No Google
    Drive access, no CUDA compilation, and no repo cloning required.

    The transformers model is downloaded via HuggingFace Hub.  Proxy
    settings are picked up automatically from the HTTPS_PROXY / https_proxy
    environment variables (TI corporate proxy is supported).

    Args:
        model_key  : Key from MODEL_CATALOG.
        output_dir : Directory to save the .onnx file.
        shape      : Custom (height, width) or None for model default.
        opset      : ONNX opset version.
        batch_size : Batch size embedded in the exported graph.
        force      : Re-export even if .onnx already exists.

    Returns:
        Absolute path of the saved .onnx file.
    """
    import torch                                        # noqa: PLC0415
    import torch.nn as nn                               # noqa: PLC0415
    from transformers import DeformableDetrForObjectDetection  # noqa: PLC0415

    info         = MODEL_CATALOG[model_key]
    hf_model_id  = info["hf_model_id"]
    export_shape = shape if shape is not None else info["shape"]
    h, w         = export_shape

    os.makedirs(output_dir, exist_ok=True)
    shape_tag = f"_{h}x{w}" if shape is not None else ""
    dst_name  = f"{model_key}{shape_tag}.onnx"
    dst_path  = os.path.join(output_dir, dst_name)

    if not force and os.path.exists(dst_path):
        print(f"[SKIP] {dst_name} already exists. Use --force to re-export.\n")
        return dst_path

    print(f"[INFO] Method       : optimum (HuggingFace transformers)")
    print(f"[INFO] HF model ID  : {hf_model_id}")
    print(f"[INFO] Input shape  : {h}×{w}  (batch {batch_size})")
    print(f"[INFO] ONNX opset   : {opset}")
    print()

    # ── Download / load from HuggingFace ─────────────────────────────────────
    print(f"[INFO] Loading model from HuggingFace …")
    print("[INFO] (First run downloads ~150–200 MB; cached at ~/.cache/huggingface/)")
    model = DeformableDetrForObjectDetection.from_pretrained(hf_model_id)
    model.eval()
    print("[INFO] Model ready.\n")

    # ── ONNX export wrapper ───────────────────────────────────────────────────
    # DeformableDetrForObjectDetection.forward(pixel_values, pixel_mask=None)
    # outputs: DeformableDetrObjectDetectionOutput with .pred_boxes and .logits
    # We rename logits → pred_logits to match our postprocess configs.
    class _HFWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, images: torch.Tensor):
            out = self.model(pixel_values=images)
            return out.pred_boxes, out.logits

    wrapper = _HFWrapper().eval()
    dummy   = torch.zeros(batch_size, 3, h, w)

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"[INFO] Exporting to ONNX (opset {opset}) …")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            dst_path,
            input_names        = ["images"],
            output_names       = ["pred_boxes", "pred_logits"],
            opset_version      = opset,
            do_constant_folding= True,
        )

    # ── Optional validation ───────────────────────────────────────────────────
    try:
        import onnx  # noqa: PLC0415
        onnx_model = onnx.load(dst_path)
        onnx.checker.check_model(onnx_model)
        print("[INFO] ONNX model validation passed.")
    except ImportError:
        pass
    except Exception as exc:
        print(f"[WARN] ONNX validation: {exc}")

    # ── Optional onnxsim simplification ──────────────────────────────────────
    if not skip_simplify:
        simplify_onnx(dst_path, force=force)

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
            "Export Deformable-DETR pretrained ONNX models.\n\n"
            "The Deformable-DETR source is cloned from GitHub on first use.\n"
            "Pretrained COCO weights are downloaded from Google Drive via gdown.\n"
            "Run --list-models to see all available variants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --method optimum                        # proxy-friendly HF download\n"
            "  %(prog)s --model deformable_detr_single_scale\n"
            "  %(prog)s --model deformable_detr_single_scale --method optimum\n"
            "  %(prog)s --model deformable_detr deformable_detr_two_stage\n"
            "  %(prog)s --model deformable_detr --shape 640 640\n"
            "  %(prog)s --model deformable_detr --weights /path/to/checkpoint.pth\n"
            "  %(prog)s --model deformable_detr --opset 18 --output-dir ./exports\n"
            "  %(prog)s --model deformable_detr --skip-simplify\n"
            "  %(prog)s --list-models"
        ),
    )

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
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help=(
            "Custom input resolution (height width). "
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
    parser.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help=(
            "Path to a local .pth checkpoint (format: {'model': state_dict, ...}). "
            "When omitted the official COCO pretrained weights are downloaded "
            "automatically from Google Drive."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        metavar="DIR",
        help=f"Directory where .onnx and .pth files will be saved. Default: {default_output}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-export and re-download even if output files already exist.",
    )
    parser.add_argument(
        "--force-reclone",
        action="store_true",
        default=False,
        help=(
            "Force re-clone of the Deformable-DETR repository, "
            "removing the cached copy in ~/.cache/deformable_detr."
        ),
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        default=False,
        help=(
            "Skip the onnxsim simplification step. "
            "By default the exported ONNX is simplified in-place with "
            "onnx-simplifier (pip install onnx-simplifier). "
            "Use this flag to skip if onnxsim is unavailable or causing issues."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress verbose progress messages.",
    )

    # ── Export method ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--method",
        choices=["torch", "optimum"],
        default="torch",
        metavar="METHOD",
        help=(
            "Export method. "
            "'torch' (default): clones the official GitHub repo and downloads "
            "weights from Google Drive via gdown. "
            "'optimum': downloads from HuggingFace Hub using the transformers "
            "library — proxy-friendly, no CUDA ops, no Google Drive required."
        ),
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

    if args.weights and len(args.model) > 1:
        print(
            "[WARN] --weights applies the same checkpoint to every model in "
            "--model.\n       This is unusual; pass a single --model variant "
            "when using custom weights."
        )

    if args.weights and args.method == "optimum":
        print("[WARN] --weights is ignored with --method optimum. "
              "HuggingFace weights are always downloaded from the Hub.\n")

    # Install dependencies appropriate to the chosen method
    if args.method == "optimum":
        ensure_hf_dependencies()
    else:
        ensure_dependencies()

    shape      = (args.shape[0], args.shape[1]) if args.shape else None
    output_dir = os.path.abspath(args.output_dir)

    exported: list[str] = []
    failed:   list[str] = []

    for model_key in args.model:
        # if MODEL_CATALOG[model_key]["dilation"]:
        #     print(
        #         f"\n[WARN] '{model_key}' is a DC5 (dilation) variant and is "
        #         "temporarily disabled because TIDL does not support dilated "
        #         "convolution in ResNet. Skipping.\n"
        #     )
        #     continue

        print(f"\n{'='*60}")
        print(f"  Exporting: {model_key}  [method={args.method}]")
        print(f"{'='*60}\n")

        try:
            if args.method == "optimum":
                out_path = export_model_optimum(
                    model_key     = model_key,
                    output_dir    = output_dir,
                    shape         = shape,
                    opset         = args.opset,
                    batch_size    = args.batch_size,
                    force         = args.force,
                    skip_simplify = args.skip_simplify,
                )
            else:
                out_path = export_model(
                    model_key     = model_key,
                    output_dir    = output_dir,
                    shape         = shape,
                    opset         = args.opset,
                    batch_size    = args.batch_size,
                    verbose       = not args.quiet,
                    custom_weights= args.weights,
                    force         = args.force,
                    force_reclone = args.force_reclone,
                    skip_simplify = args.skip_simplify,
                )
            exported.append(out_path)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[ERROR] Export failed for '{model_key}': {exc}")
            import traceback
            traceback.print_exc()
            failed.append(model_key)

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

    if failed and args.method == "torch":
        failed_str = " ".join(failed)
        print(
            "[TIP] The torch method failed (common causes: Google Drive blocked\n"
            "      by a corporate proxy, or missing CUDA ops).\n"
            "      Try the HuggingFace-based export instead — it downloads from\n"
            "      HuggingFace Hub and requires no Google Drive access:\n"
            f"\n"
            f"        python prepare_model.py --method optimum --model {failed_str}\n"
        )
    elif failed and args.method == "optimum":
        failed_str = " ".join(failed)
        print(
            "[TIP] The optimum method failed.\n"
            "      If HuggingFace Hub is accessible, check your transformers\n"
            "      installation.  You can also try the torch method with a\n"
            "      manually downloaded checkpoint:\n"
            f"\n"
            f"        python prepare_model.py --method torch --model {failed_str}\n"
            f"        python prepare_model.py --method torch --model {failed_str} "
            f"--weights /path/to/checkpoint.pth\n"
        )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
