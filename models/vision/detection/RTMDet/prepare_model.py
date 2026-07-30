#!/usr/bin/env python3
"""
Script to download RTMDet models (from .link files) and convert them to ONNX.
Also capable of generating .link files for multiple variants.
Includes ONNX shape inference, simplification, and batch size fixing (like YOLOX script).
"""

import os
import sys
import subprocess
import re
import importlib


def pip_install(packages: list[str], label: str = "") -> None:
    """
    Run pip install for the given list of package specs.
    Prints captured output only on failure so the user can diagnose.

    Args:
        packages : List of pip install specs (e.g. ['mmcv-lite<2.2.0', 'torch']).
        label    : Human-readable description for log messages.
    """
    tag = f"[{label}]" if label else "[DEP]"
    print(f"{tag} Installing: {', '.join(packages)} …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"{tag} ERROR: pip install failed (exit code {result.returncode}).")
        print(f"{tag} --- pip output ---")
        print(result.stdout)
        print(f"{tag} --- end pip output ---")
        sys.exit(1)
    print(f"{tag} Done.\n")


def ensure_mmcv() -> None:
    """
    Ensure mmcv-lite<2.2.0 is installed.
    mmdet requires mmcv>=2.0.0rc4,<2.2.0. We use mmcv-lite (pure-Python wheel,
    no C++ compilation) which satisfies this requirement.

    If an incompatible version (>=2.2.0) is already installed, it is removed first.
    This must be called BEFORE mmdet is installed so mmdet doesn't pull in 2.2.0.
    """
    try:
        import mmcv
        version = mmcv.__version__
        major, minor = map(int, version.split('.')[:2])
        if major > 2 or (major == 2 and minor >= 2):
            print(f"[DEP] mmcv {version} is incompatible (need <2.2.0) – removing …")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-lite", "mmcv-full"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            raise ImportError  # fall through to install
        print(f"[DEP]  ✔  mmcv {version} is compatible.")
    except ImportError:
        pip_install(["mmcv-lite<2.2.0"], label="DEP")


def check_dependencies(packages: dict[str, str]) -> None:
    """
    Check that every package in *packages* can be imported.
    Any package that is missing is installed automatically via pip.

    Args:
        packages : Mapping of { import_name: pip_install_name }.
    """
    missing: list[str] = []

    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
            print(f"[DEP]  ✔  {import_name} is already installed.")
        except ImportError:
            print(f"[DEP]  ✘  {import_name} not found – will install '{pip_name}'.")
            missing.append(pip_name)

    if not missing:
        print("[DEP] All dependencies satisfied.\n")
        return

    pip_install(missing)


def download_file(url: str, save_path: str, label: str = "file") -> bool:
    """
    Download a single file from *url* to *save_path*.

    Returns True on success, False on failure (does NOT call sys.exit so the
    caller can decide whether to fall back to an alternative).

    Args:
        url       : HTTP/HTTPS URL of the file to download.
        save_path : Destination path (directories are created automatically).
        label     : Human-readable name used in log messages.
    """
    # Create the destination directory if it does not already exist
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Skip download if the file already exists
    if os.path.exists(save_path):
        print(f"[INFO] {label} already exists at: {save_path}")
        return True

    print(f"[INFO] Downloading {label} …")
    print(f"       URL  : {url}")
    print(f"       Dest : {save_path}")
    print()

    try:
        import requests
        from tqdm import tqdm
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = len(data)
                file.write(data)
                bar.update(size)
        print(f"[SUCCESS] {label} saved to: {save_path}\n")
        return True
    except Exception as exc:
        # Remove any partial file so it is not mistaken for a complete download
        if os.path.exists(save_path):
            os.remove(save_path)
        print(f"\n[WARN] Could not download {label}: {exc}")
        return False


# mmdetection GitHub tag from which configs are downloaded
_MMDET_TAG = "v3.3.0"
_GITHUB_BASE = f"https://raw.githubusercontent.com/open-mmlab/mmdetection/{_MMDET_TAG}/configs"

# All config files required to load any RTMDet variant, with their relative
# paths inside a local "configs/" directory mirroring the GitHub layout.
_CONFIG_FILES = {
    "rtmdet/rtmdet_l_8xb32-300e_coco.py":    f"{_GITHUB_BASE}/rtmdet/rtmdet_l_8xb32-300e_coco.py",
    "rtmdet/rtmdet_m_8xb32-300e_coco.py":    f"{_GITHUB_BASE}/rtmdet/rtmdet_m_8xb32-300e_coco.py",
    "rtmdet/rtmdet_s_8xb32-300e_coco.py":    f"{_GITHUB_BASE}/rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "rtmdet/rtmdet_tiny_8xb32-300e_coco.py": f"{_GITHUB_BASE}/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    "rtmdet/rtmdet_x_8xb32-300e_coco.py":    f"{_GITHUB_BASE}/rtmdet/rtmdet_x_8xb32-300e_coco.py",
    "rtmdet/rtmdet_tta.py":                   f"{_GITHUB_BASE}/rtmdet/rtmdet_tta.py",
    "_base_/default_runtime.py":              f"{_GITHUB_BASE}/_base_/default_runtime.py",
    "_base_/schedules/schedule_1x.py":        f"{_GITHUB_BASE}/_base_/schedules/schedule_1x.py",
    "_base_/datasets/coco_detection.py":      f"{_GITHUB_BASE}/_base_/datasets/coco_detection.py",
}

# Config path (relative to configs/) for each variant
_VARIANT_CONFIG = {
    "tiny": "rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    "s":    "rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "m":    "rtmdet/rtmdet_m_8xb32-300e_coco.py",
    "l":    "rtmdet/rtmdet_l_8xb32-300e_coco.py",
    "x":    "rtmdet/rtmdet_x_8xb32-300e_coco.py",
}

CONFIGS_DIR = "configs"


def setup_configs() -> None:
    """
    Download all mmdetection config files needed to load RTMDet variants.

    Files are stored under ./configs/ mirroring the GitHub layout so that
    _base_ relative-path references resolve correctly:
        configs/
          _base_/
            default_runtime.py
            schedules/schedule_1x.py
            datasets/coco_detection.py
          rtmdet/
            rtmdet_l_8xb32-300e_coco.py   ← _base_ = ../_base_/...
            rtmdet_s_8xb32-300e_coco.py   ← _base_ = ./rtmdet_l...
            rtmdet_m_8xb32-300e_coco.py
            rtmdet_tiny_8xb32-300e_coco.py
            rtmdet_tta.py
    """
    print("[CONFIG] Downloading mmdetection config files …")
    for rel_path, url in _CONFIG_FILES.items():
        dest = os.path.join(CONFIGS_DIR, rel_path)
        download_file(url, dest, label=rel_path)
    print("[CONFIG] All config files ready.\n")


def get_config_path(variant: str) -> str:
    """Return the local filesystem path to the config file for *variant*."""
    rel = _VARIANT_CONFIG[variant]
    return os.path.join(CONFIGS_DIR, rel)


def get_input_shape_from_config(config_path: str) -> tuple[int, int]:
    """
    Extract input shape from config file.
    Looks for img_scale or similar parameters.
    Returns (height, width) to match PyTorch NCHW convention.
    Defaults to (640, 640) if not found.
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Look for common patterns - most configs specify (width, height)
        patterns = [
            r'img_scale\s*=\s*\((\d+),\s*(\d+)\)',
            r'width\s*=\s*(\d+).*?height\s*=\s*(\d+)',
            r'input_size\s*=\s*\((\d+),\s*(\d+)\)',
            r'img_scale\s*=\s*\[(\d+),\s*(\d+)\]',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                first, second = map(int, match.groups())
                # Most configs use (width, height), convert to (height, width) for PyTorch
                return (second, first)
    except Exception as e:
        print(f"Warning: Could not read config file {config_path}: {e}")

    # Default input shape for RTMDet (height, width)
    return (640, 640)


def fix_onnx_batch_size(onnx_path: str, batch_size: int = 1) -> None:
    """
    Post-process an ONNX model to hard-code the batch dimension to *batch_size*.

    Some exporters (including torch.onnx.export) may leave the first dimension
    of inputs/outputs as a symbolic string (e.g. 'batch') even when
    dynamic_axes is not specified.  This function:

      1. Loads the ONNX protobuf from *onnx_path*.
      2. Iterates over every input and output in the graph.
      3. Replaces the first dimension with the integer *batch_size*.
      4. Re-runs ONNX shape inference so downstream tools see correct shapes.
      5. Overwrites *onnx_path* with the fixed model.

    Args:
        onnx_path  : Path to the ONNX file to fix (modified in-place).
        batch_size : Integer value to set for the batch dimension (default 1).
    """
    import onnx                          # noqa: PLC0415
    import onnx.shape_inference          # noqa: PLC0415

    print(f"[POST] Fixing batch dimension to {batch_size} in: {onnx_path}")

    # Load the model from disk
    model_proto = onnx.load(onnx_path)
    graph = model_proto.graph

    # ── Fix inputs ─────────────────────────────────────────────────────────
    for tensor in graph.input:
        shape = tensor.type.tensor_type.shape
        if shape.dim:
            dim = shape.dim[0]
            # Clear any symbolic name (e.g. "batch") and set the integer value
            dim.ClearField("dim_param")
            dim.dim_value = batch_size

    # ── Fix outputs ────────────────────────────────────────────────────────
    for tensor in graph.output:
        shape = tensor.type.tensor_type.shape
        if shape.dim:
            dim = shape.dim[0]
            dim.ClearField("dim_param")
            dim.dim_value = batch_size

    # Re-run shape inference so the rest of the graph reflects the fixed shape
    model_proto = onnx.shape_inference.infer_shapes(model_proto)

    # Overwrite the original file with the fixed model
    onnx.save(model_proto, onnx_path)
    print(f"[POST] Batch dimension fixed → shape now starts with {batch_size}.\n")


def _stub_mmcv_ext() -> None:
    """
    Stub out mmcv._ext so mmdet can be imported when only mmcv-lite is installed.

    mmcv-lite is the pure-Python variant of mmcv and does not ship the compiled
    C++ extension (_ext).  mmdet's import chain pulls in mmcv.ops, which eagerly
    loads _ext even though the ops themselves are not needed for a forward-pass /
    ONNX export.  Installing the full mmcv requires nvcc, which may not be
    present.

    The stub raises a clear RuntimeError only if one of the missing ops is
    *actually called* at runtime, so normal ONNX tracing is unaffected.
    """
    import sys, types

    if 'mmcv._ext' in sys.modules:
        return  # already present (real or stubbed)

    import importlib.machinery

    def _make_stub_op(name):
        def _op(*a, **kw):
            raise RuntimeError(
                f"mmcv._ext.{name} was called but mmcv-lite is installed "
                "(no C++ extensions). Install full mmcv if this op is required."
            )
        _op.__name__ = name
        return _op

    # Build the stub module with a valid (non-None) __spec__ so that
    # pkgutil.find_loader('mmcv._ext') succeeds and returns None (loader=None),
    # causing mmengine.mmcv_full_available() to correctly return False.
    stub = types.ModuleType('mmcv._ext')
    stub.__file__ = '<mmcv._ext stub>'
    stub.__package__ = 'mmcv'
    stub.__path__ = []
    stub.__loader__ = None
    stub.__spec__ = importlib.machinery.ModuleSpec(
        'mmcv._ext', loader=None, origin='<mmcv._ext stub>'
    )

    stub.__class__ = type(
        '_StubExtModule',
        (types.ModuleType,),
        {'__getattr__': lambda self, name: _make_stub_op(name)},
    )
    sys.modules['mmcv._ext'] = stub


def convert_to_onnx(pth_path: str, config_path: str, onnx_path: str, input_shape: tuple[int, int] = (640, 640), simplify: bool = False) -> bool:
    """
    Convert PyTorch model to ONNX using MMDetection tools.
    Includes shape inference, batch size fixing, and optional simplification.

    Args:
        input_shape: (height, width) tuple for input dimensions
    """
    # Stub mmcv._ext before importing mmdet so the import chain succeeds
    # even when only mmcv-lite (no compiled C++ extensions) is installed.
    _stub_mmcv_ext()

    try:
        from mmdet.apis import init_detector
        import torch
    except ImportError as e:
        print(f"Error importing MMDetection: {e}")
        print("Please ensure mmdet and mmcv are installed.")
        return False

    # PyTorch >=2.6 defaults torch.load to weights_only=True, which blocks
    # several legacy objects (HistoryBuffer, numpy arrays) embedded in mmengine
    # checkpoints. We patch torch.load to use weights_only=False for the
    # duration of init_detector. The checkpoint is from a trusted source
    # (download.openmmlab.com), so this is safe.
    import warnings
    import io
    import contextlib

    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': False})

    # Suppress two known harmless messages from init_detector:
    #  1. FutureWarning: torch.cuda.amp.autocast deprecated — issued via
    #     typing_extensions @deprecated on __init__, needs a persistent filter
    #     (not catch_warnings) because stacklevel points into mmdet modules.
    #  2. "Loads checkpoint / unexpected key" — MMLogger writes to stdout;
    #     redirect_stdout captures it cleanly.
    warnings.filterwarnings('ignore', category=FutureWarning)

    try:
        _captured = io.StringIO()
        with contextlib.redirect_stdout(_captured):
            model = init_detector(config_path, pth_path, device='cpu')
        model.eval()

        # Create dummy input - PyTorch uses (B, C, H, W) format
        height, width = input_shape
        dummy_input = torch.randn(1, 3, height, width)

        # Build an export wrapper that includes the full post-processing decode:
        #   1. Backbone + neck + head  (mode='tensor' — raw logits + ltrb distances)
        #   2. Sigmoid on cls scores
        #   3. distance2bbox decode using pre-computed anchor center points (priors)
        #
        # NMS is intentionally omitted — it requires dynamic output shapes that are
        # not supported in ONNX without custom ops, and is best run on-device after
        # inference.
        #
        # Output:
        #   boxes  : (1, N, 4)   xyxy pixel coordinates (unclipped)
        #   scores : (1, N, C)   per-class sigmoid probabilities
        # where N = sum(H_i * W_i) over FPN levels = 8400 for a 640×640 input.
        class _RTMDetDecodeWrapper(torch.nn.Module):
            def __init__(self, m, input_h: int, input_w: int):
                super().__init__()
                self.m = m
                # Pre-compute anchor center points for all FPN levels.
                # Strides come from the prior generator; default RTMDet uses [8, 16, 32].
                strides = [s[0] for s in m.bbox_head.prior_generator.strides]
                priors_list = []
                for stride in strides:
                    feat_h = input_h // stride
                    feat_w = input_w // stride
                    shift_x = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * stride
                    shift_y = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * stride
                    yy, xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
                    priors_list.append(
                        torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
                    )
                # priors: (N_total, 2) — constant for a fixed input size
                self.register_buffer('priors', torch.cat(priors_list, dim=0))

            def forward(self, img):
                feats = self.m.extract_feat(img)
                cls_scores, bbox_preds = self.m.bbox_head(feats)

                all_scores, all_dists = [], []
                for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
                    B = cls_score.shape[0]
                    # (B, C, H, W) -> (B, H*W, C)
                    all_scores.append(
                        cls_score.permute(0, 2, 3, 1).reshape(B, -1, cls_score.shape[1])
                    )
                    # (B, 4, H, W) -> (B, H*W, 4)
                    all_dists.append(
                        bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
                    )

                scores = torch.cat(all_scores, dim=1).sigmoid()  # (B, N, C)
                dists  = torch.cat(all_dists,  dim=1)            # (B, N, 4)

                # distance2bbox: (cx - dl, cy - dt, cx + dr, cy + db)
                priors = self.priors.unsqueeze(0)                # (1, N, 2)
                x1 = priors[..., 0] - dists[..., 0]
                y1 = priors[..., 1] - dists[..., 1]
                x2 = priors[..., 0] + dists[..., 2]
                y2 = priors[..., 1] + dists[..., 3]
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)   # (B, N, 4)

                return boxes, scores

        export_model = _RTMDetDecodeWrapper(model, height, width)
        export_model.eval()

        # Dry run to confirm output shapes.
        with torch.no_grad():
            boxes_out, scores_out = export_model(dummy_input)
        print(f"[CONV] boxes : {list(boxes_out.shape)}")   # (1, 8400, 4)
        print(f"[CONV] scores: {list(scores_out.shape)}")  # (1, 8400, 80)

        # Export to ONNX — two named outputs: boxes (xyxy) and scores (per-class sigmoid)
        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['boxes', 'scores'],
        )
        print(f"[CONV] Raw ONNX written to: {onnx_path}")
        
        # Fix batch size to 1 (explicitly)
        fix_onnx_batch_size(onnx_path, batch_size=1)
        
        # Run shape inference
        try:
            import onnx
            import onnx.shape_inference
            print("[POST] Running ONNX shape inference …")
            model_proto = onnx.load(onnx_path)
            model_proto = onnx.shape_inference.infer_shapes(model_proto)
            onnx.save(model_proto, onnx_path)
            print("[POST] Shape inference complete – model saved.\n")
        except ImportError:
            print("[POST] WARNING: 'onnx' package not found – skipping shape inference.\n")
        except Exception as exc:
            print(f"[POST] WARNING: shape inference failed ({exc}) – model unchanged.\n")
        
        # Optional ONNX simplification
        if simplify:
            try:
                import onnx
                import onnxsim
                print("[SIMPLIFY] Simplifying model …")
                model = onnx.load(onnx_path)
                model_simplified, check = onnxsim.simplify(
                    model,
                    check_n=3,
                    perform_optimization=True,
                    skip_fuse_bn=False,
                )
                if not check:
                    print("[SIMPLIFY] Warning: Simplification validation failed")
                else:
                    print("[SIMPLIFY] Simplification successful and validated")
                onnx.save(model_simplified, onnx_path)
                print(f"[SIMPLIFY] Simplified model saved to: {onnx_path}\n")
            except ImportError:
                print("[SIMPLIFY] WARNING: 'onnx-simplifier' package not found – skipping simplification.\n")
            except Exception as exc:
                print(f"[SIMPLIFY] WARNING: simplification failed ({exc}) – using original model.\n")
        
        print(f"[SUCCESS] ONNX model saved to: {onnx_path}")
        return True
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return False
    finally:
        torch.load = _orig_torch_load  # always restore, even on exception


def process_link_file(link_file_path: str, simplify: bool = False) -> bool:
    """
    Process a single .link file: download PTH and convert to ONNX.

    Expected .link format (single line):
        <pth_url> -o <onnx_filename>
    """
    with open(link_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        print(f"Warning: {link_file_path} is empty.")
        return False

    parts = lines[0].split()
    if len(parts) < 3 or parts[-2] != '-o':
        print(f"Warning: Unexpected format in {link_file_path}: {lines[0]}")
        return False

    pth_url = parts[0]
    onnx_filename = parts[-1]

    # Derive variant from filename: "rtmdet_tiny.onnx" → "tiny"
    model_name = os.path.splitext(onnx_filename)[0]   # rtmdet_tiny
    variant = model_name.split('_')[1]                 # tiny
    pth_filename = model_name + '.pth'

    if variant not in _VARIANT_CONFIG:
        print(f"[WARN] Cannot determine variant from filename '{onnx_filename}'. Skipping.")
        return False

    # Download checkpoint
    if not os.path.exists(pth_filename):
        if not download_file(pth_url, pth_filename, label=f"RTMDet-{variant} checkpoint"):
            print(f"Failed to download checkpoint. Skipping conversion.")
            return False
    else:
        print(f"[INFO] Checkpoint already exists: {pth_filename}")

    # Resolve config path (downloaded by setup_configs)
    config_path = get_config_path(variant)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}. Run setup_configs() first.")
        return False

    input_shape = get_input_shape_from_config(config_path)
    print(f"[INFO] Input shape: {input_shape}")

    success = convert_to_onnx(pth_filename, config_path, onnx_filename, input_shape, simplify=simplify)
    if success:
        print(f"[SUCCESS] Conversion complete: {onnx_filename}")
    else:
        print(f"[ERROR] Conversion failed for {onnx_filename}")

    return success


def generate_link_files(variants=None, force=False):
    """
    Generate .link files for RTMDet variants with correct model URLs.

    Args:
        variants: List of variant names. If None, uses default list.
        force: If True, overwrite existing .link files.
    """
    if variants is None:
        variants = ['tiny', 's', 'm', 'l']

    base_url = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet"

    # Actual model URLs with correct hashes (verified from OpenMMLab)
    # Format: variant -> (model_dir, checkpoint_filename)
    model_info = {
        'tiny': ('rtmdet_tiny_8xb32-300e_coco', 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'),
        's':    ('rtmdet_s_8xb32-300e_coco',    'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'),
        'm':    ('rtmdet_m_8xb32-300e_coco',    'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'),
        'l':    ('rtmdet_l_8xb32-300e_coco',    'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'),
        'x':    ('rtmdet_x_8xb32-300e_coco',    'rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'),
    }

    for variant in variants:
        if variant not in model_info:
            print(f"[WARN] Unknown variant '{variant}' - skipping")
            continue

        model_dir, pth_filename = model_info[variant]
        pth_url = f"{base_url}/{model_dir}/{pth_filename}"
        onnx_filename = f"rtmdet_{variant}.onnx"
        link_content = f"{pth_url} -o {onnx_filename}\n"

        link_file = f"rtmdet_{variant}.onnx.link"

        # Skip if file exists and force=False
        if os.path.exists(link_file) and not force:
            print(f"[INFO] {link_file} already exists (use --generate-links to regenerate)")
            continue

        with open(link_file, 'w') as f:
            f.write(link_content)
        print(f"[INFO] Generated {link_file}")


KNOWN_VARIANTS = ['tiny', 's', 'm', 'l', 'x']


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Download RTMDet models and convert them to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                        # download all models\n"
            "  %(prog)s --models tiny s        # download only tiny and s\n"
            "  %(prog)s --models l --simplify\n"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        choices=KNOWN_VARIANTS,
        default=None,
        help=(
            f"One or more model variants to download and convert. "
            f"Choices: {KNOWN_VARIANTS}. "
            f"Defaults to all variants if omitted."
        ),
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify the ONNX model using onnx-simplifier after conversion.",
    )
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="Do not simplify the ONNX model.",
    )
    parser.add_argument(
        "--generate-links",
        action="store_true",
        help="Force regeneration of .link files even if they already exist.",
    )
    args = parser.parse_args()

    variants = args.models if args.models else KNOWN_VARIANTS
    print(f"Models selected: {variants}\n")

    print("Checking dependencies...")

    # Stub mmcv._ext early so that importing mmdet during the dependency check
    # does not fail with "No module named 'mmcv._ext'" when mmcv-lite is installed.
    _stub_mmcv_ext()

    # mmcv must be installed before mmdet to prevent mmdet pulling in mmcv>=2.2.0
    ensure_mmcv()

    check_dependencies({
        'torch': 'torch',
        'mmdet': 'mmdet',
        'requests': 'requests',
        'tqdm': 'tqdm',
        'onnx': 'onnx',
        'onnxsim': 'onnxsim',
    })

    # Generate link files only for the selected models
    print("\nSetting up link files for selected models...")
    generate_link_files(variants, force=args.generate_links)

    # Download all mmdetection config files (shared across all variants)
    setup_configs()

    # Process link files for the selected models only
    print("\nProcessing .link files...")
    link_files = sorted([
        f"rtmdet_{v}.onnx.link"
        for v in variants
        if os.path.exists(f"rtmdet_{v}.onnx.link")
    ])
    if not link_files:
        print("No .link files found.")
        return

    for link_file in link_files:
        print(f"\n{'='*60}")
        print(f"Processing {link_file}")
        print('='*60)
        process_link_file(link_file, simplify=args.simplify)

    print("\n" + "="*60)
    print(f"Done! Converted {len(link_files)} model(s): {', '.join(v for v in variants)}")
    print("="*60)


if __name__ == "__main__":
    main()