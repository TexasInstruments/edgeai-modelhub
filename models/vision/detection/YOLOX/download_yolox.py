"""Script to obtain YOLOX ONNX model(s).

Strategy (tried in order for each model):
  1. Download the pre-built ONNX directly from GitHub releases.
  2. If the ONNX download fails, download the PyTorch weights (.pth) and
     convert them to ONNX locally using the official YOLOX export utility.

After the ONNX model is obtained (either downloaded or converted), an optional
accuracy-verification step runs the model against the COCO val2017 dataset and
reports mAP@[0.50:0.95] and mAP@0.50 using pycocotools.

Supported model names (pass via --model or edit MODEL_NAME below):
  yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l, yolox_x, yolox_darknet53

Model source : https://github.com/Megvii-BaseDetection/YOLOX
ONNX release : v0.1.1rc0
PTH  release : 0.0.1 (storage repo)

Usage:
  python download_yolox.py --model yolox_nano
  python download_yolox.py --model yolox_s yolox_m
  python download_yolox.py --model yolox_nano yolox_s --verify
  python download_yolox.py --model yolox_l --num-val-images 50
  python download_yolox.py --model yolox_x yolox_darknet53 --verify --coco-dir /path/to/coco
  python download_yolox.py --model yolox_nano --simplify
  python download_yolox.py --model yolox_s yolox_m --verify --simplify
"""

import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


# ─────────────────────────────────────────────
# .link file reader
# ─────────────────────────────────────────────

def read_url_from_link_file(link_path: str) -> str:
    """
    Read the download URL from a *.link file.

    The file format is a single line:
        <url> -o <output_filename>

    Only the URL (first whitespace-delimited token) is returned.

    Args:
        link_path : Path to the .link file (relative or absolute).

    Returns:
        The URL string extracted from the file.

    Raises:
        FileNotFoundError : If *link_path* does not exist.
        ValueError        : If the file is empty or has no URL token.
    """
    if not os.path.exists(link_path):
        raise FileNotFoundError(f"[LINK] .link file not found: {link_path}")

    with open(link_path, "r") as fh:
        line = fh.readline().strip()

    if not line:
        raise ValueError(f"[LINK] .link file is empty: {link_path}")

    url = line.split()[0]
    return url


# ─────────────────────────────────────────────
# Dependency checker / auto-installer
# ─────────────────────────────────────────────

# Map of  import-name  →  pip-install-name
# Standard-library modules do NOT need to be listed here.
# torch / onnx are only needed for the .pth → .onnx conversion fallback;
# they are added dynamically inside convert_pth_to_onnx() if required.
REQUIRED_PACKAGES: dict[str, str] = {
    "onnxsim": "onnx-simplifier",
}

# Default model name – override via --model CLI argument or by editing this value.
MODEL_NAME = "yolox_s"

# Map from model_name (underscore form) to the experiment name used by YOLOX's
# get_exp() API (hyphen form).  Extend this dict when new variants are released.
MODEL_EXP_NAME: dict[str, str] = {
    "yolox_nano":       "yolox-nano",
    "yolox_tiny":       "yolox-tiny",
    "yolox_s":          "yolox-s",
    "yolox_m":          "yolox-m",
    "yolox_l":          "yolox-l",
    "yolox_x":          "yolox-x",
    "yolox_darknet53":  "yolov3",
}

# Default input resolution per model variant (height, width).
# YOLOX Nano / Tiny use 416; all others use 640.
MODEL_INPUT_SIZE: dict[str, tuple[int, int]] = {
    "yolox_nano":       (416, 416),
    "yolox_tiny":       (416, 416),
    "yolox_s":          (640, 640),
    "yolox_m":          (640, 640),
    "yolox_l":          (640, 640),
    "yolox_x":          (640, 640),
    "yolox_darknet53":  (640, 640),
}


def ensure_dependencies(packages: dict[str, str]) -> None:
    """
    Check that every package in *packages* can be imported.
    Any package that is missing is installed automatically via pip.

    Args:
        packages : Mapping of  { import_name: pip_install_name }.
                   Use the *import* name as the key (e.g. "PIL") and the
                   *pip* name as the value (e.g. "Pillow").
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
        if packages:
            print("[DEP] All dependencies satisfied.\n")
        return

    print(f"\n[DEP] Installing missing packages: {', '.join(missing)} …")
    try:
        # Use the same Python interpreter that is running this script
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "--no-build-isolation"],
            stdout=subprocess.DEVNULL,   # suppress pip's verbose output
            stderr=subprocess.STDOUT,
        )
        print("[DEP] Installation complete.\n")
    except subprocess.CalledProcessError as exc:
        print(f"[DEP] ERROR: pip install failed (exit code {exc.returncode}).")
        print("[DEP] Please install the missing packages manually and re-run.")
        sys.exit(1)


# ─────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────

def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """
    Callback used by urllib.request.urlretrieve to display download progress.

    Args:
        block_num  : Number of blocks transferred so far.
        block_size : Size of each block in bytes.
        total_size : Total size of the file in bytes (-1 if unknown).
    """
    if total_size > 0:
        downloaded = block_num * block_size
        # Clamp to 100 % in case the last block overshoots
        percent = min(downloaded / total_size * 100, 100.0)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        # \r rewrites the same line so the terminal stays clean
        sys.stdout.write(
            f"\r  Downloading: {percent:5.1f}%  "
            f"({downloaded_mb:.2f} MB / {total_mb:.2f} MB)"
        )
        sys.stdout.flush()
    else:
        # Total size unknown – just show bytes downloaded
        downloaded_mb = (block_num * block_size) / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {downloaded_mb:.2f} MB")
        sys.stdout.flush()


# ─────────────────────────────────────────────
# Generic file downloader
# ─────────────────────────────────────────────

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
        urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
        print()  # newline after progress bar
        print(f"[SUCCESS] {label} saved to: {save_path}\n")
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
        # Remove any partial file so it is not mistaken for a complete download
        if os.path.exists(save_path):
            os.remove(save_path)
        print(f"\n[WARN] Could not download {label}: {exc}")
        return False


# ─────────────────────────────────────────────
# YOLOX source installer (git-based)
# ─────────────────────────────────────────────

# Directory where the YOLOX repo will be cloned if pip install fails
YOLOX_CLONE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_yolox_src")

# Official YOLOX GitHub repository URL
YOLOX_REPO_URL = "https://github.com/Megvii-BaseDetection/YOLOX.git"


def ensure_yolox() -> None:
    """
    Make the 'yolox' package importable using the best available method:

      1. Already importable  → nothing to do.
      2. pip install via git URL  → fast, installs into site-packages.
      3. git clone + sys.path injection  → fallback when pip/git-pip fails
         (e.g. no git credential, corporate proxy).  The repo is cloned to
         YOLOX_CLONE_DIR next to this script and added to sys.path so that
         'import yolox' resolves correctly.
    """
    # ── Check 1: already importable ────────────────────────────────────────
    try:
        importlib.import_module("yolox")
        print("[DEP]  ✔  yolox is already importable.")
        return
    except ImportError:
        pass

    # ── Check 2: try pip install from GitHub ───────────────────────────────
    git_pip_url = f"git+{YOLOX_REPO_URL}"
    print(f"[DEP]  ✘  yolox not found – trying: pip install {git_pip_url}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", git_pip_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        # Verify the install actually worked
        importlib.import_module("yolox")
        print("[DEP] yolox installed via pip (git URL).\n")
        return
    except (subprocess.CalledProcessError, ImportError):
        print("[DEP] pip git-install failed – falling back to git clone …")

    # ── Check 3: git clone fallback ────────────────────────────────────────
    if not shutil.which("git"):
        print(
            "[DEP] ERROR: 'git' executable not found on PATH.\n"
            "      Please install git or manually run:\n"
            f"        pip install git+{YOLOX_REPO_URL}"
        )
        sys.exit(1)

    # Remove a stale / incomplete clone if present
    if os.path.exists(YOLOX_CLONE_DIR):
        print(f"[DEP] Removing stale clone at {YOLOX_CLONE_DIR} …")
        shutil.rmtree(YOLOX_CLONE_DIR)

    print(f"[DEP] Cloning YOLOX repository to {YOLOX_CLONE_DIR} …")
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", YOLOX_REPO_URL, YOLOX_CLONE_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[DEP] ERROR: git clone failed (exit code {exc.returncode}).")
        sys.exit(1)

    # Install the cloned package's requirements so the import works fully
    req_file = os.path.join(YOLOX_CLONE_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("[DEP] Installing YOLOX requirements …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    # Add the cloned repo root to sys.path so 'import yolox' resolves
    if YOLOX_CLONE_DIR not in sys.path:
        sys.path.insert(0, YOLOX_CLONE_DIR)

    # Final verification
    try:
        importlib.import_module("yolox")
        print("[DEP] yolox is now importable via cloned source.\n")
    except ImportError:
        print(
            "[DEP] ERROR: yolox still not importable after cloning.\n"
            "      Please report this issue or install manually."
        )
        sys.exit(1)


# ─────────────────────────────────────────────
# ONNX batch-size fixer
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# PTH → ONNX conversion
# ─────────────────────────────────────────────

def convert_pth_to_onnx(
    pth_path: str,
    onnx_path: str,
    model_name: str,
    input_size: tuple[int, int],
) -> None:
    """
    Convert a YOLOX PyTorch checkpoint (.pth) to ONNX format.

    This function:
      1. Ensures torch, onnx, and yolox are available (auto-installs if needed).
      2. Loads the YOLOX model architecture for the given *model_name*.
      3. Loads the checkpoint weights.
      4. Exports the model to ONNX using torch.onnx.export.

    Args:
        pth_path   : Path to the downloaded .pth checkpoint file.
        onnx_path  : Destination path for the exported .onnx file.
        model_name : YOLOX variant name (e.g. "yolox_nano", "yolox_s").
                     Must be a key in MODEL_EXP_NAME.
        input_size : (height, width) of the model's expected input image.
    """

    # ── Step A: ensure torch, onnx, and onnxscript are installed ──────────
    # onnxscript is required by torch >= 2.1's ONNX exporter internals.
    print("[CONV] Checking conversion dependencies …")
    ensure_dependencies({
        "torch":      "torch",
        "onnx":       "onnx",
        "onnxscript": "onnxscript",   # needed by torch.onnx internals (torch >= 2.1)
    })

    # ── Step B: ensure yolox is importable (git-aware installer) ──────────
    ensure_yolox()

    # ── Step C: import after installation ──────────────────────────────────
    import torch  # noqa: PLC0415  (import inside function is intentional)

    # Import YOLOX experiment / model builder
    from yolox.exp import get_exp  # noqa: PLC0415

    # ── Step D: build the YOLOX model ──────────────────────────────────────
    exp_name = MODEL_EXP_NAME.get(model_name)
    if exp_name is None:
        print(
            f"[CONV] ERROR: unknown model '{model_name}'.\n"
            f"       Known models: {', '.join(MODEL_EXP_NAME)}"
        )
        sys.exit(1)

    print(f"[CONV] Building {model_name} model architecture (exp: {exp_name}) …")
    exp = get_exp(exp_name=exp_name)
    model = exp.get_model()
    model.eval()

    # ── Step E: load checkpoint weights ────────────────────────────────────
    print(f"[CONV] Loading weights from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location="cpu")

    # YOLOX checkpoints may wrap weights under a 'model' key
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print("[CONV] Weights loaded successfully.\n")

    # ── Step F: export to ONNX ─────────────────────────────────────────────
    print(f"[CONV] Exporting to ONNX (input size {input_size[0]}×{input_size[1]}) …")

    # Dummy input tensor: batch=1, channels=3, H, W
    dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    # Use the legacy TorchScript-based exporter explicitly.
    # torch >= 2.1 introduced a new dynamo-based exporter that requires
    # 'onnxscript'; passing dynamo=False forces the stable legacy path
    # which works with any torch version and avoids the onnxscript dependency.
    #
    # NOTE: dynamic_axes is intentionally omitted here so the exporter
    # traces with a fixed batch=1.  The post-processing step below then
    # hard-codes the batch dimension in the ONNX graph's shape info to
    # guarantee runtimes see [1, 3, H, W] instead of [batch, 3, H, W].
    export_kwargs: dict = dict(
        opset_version=18,           # opset 11 is widely supported by runtimes
        input_names=["images"],
        output_names=["output"],
    )

    # dynamo=False is only accepted by torch >= 2.1; guard with inspect so
    # the script also works on older torch versions.
    import inspect  # noqa: PLC0415
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False  # force legacy TorchScript exporter

    torch.onnx.export(model, dummy_input, onnx_path, **export_kwargs)
    print(f"[CONV] Raw ONNX written to: {onnx_path}")

    # ── Step G: fix batch dimension to 1 in the ONNX graph ────────────────
    # Even when dynamic_axes is omitted, some exporters still emit a symbolic
    # 'batch' dim.  This step loads the graph and explicitly overwrites the
    # first dimension of every input and output tensor to the integer 1.
    fix_onnx_batch_size(onnx_path, batch_size=1)

    print(f"[SUCCESS] ONNX model (batch=1) saved to: {onnx_path}\n")


# ─────────────────────────────────────────────
# COCO val2017 accuracy verification
# ─────────────────────────────────────────────

# COCO val2017 image archive and annotation URLs (official mirrors)
COCO_VAL_IMAGES_URL  = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_ANNOTS_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category IDs in the order YOLOX was trained on (80-class subset).
# These map the 0-based class index produced by the model to the official
# COCO category_id values expected by pycocotools.
COCO80_CATEGORY_IDS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def _letterbox(
    img: "np.ndarray",
    target_h: int,
    target_w: int,
) -> tuple["np.ndarray", float]:
    """
    Resize *img* to fit inside a (target_h × target_w) canvas while preserving
    the aspect ratio.  The canvas is filled with grey (114, 114, 114).

    Returns:
        padded_img : uint8 array of shape (target_h, target_w, 3).
        ratio      : scale factor applied to the original image dimensions.
    """
    import numpy as np  # noqa: PLC0415

    h0, w0 = img.shape[:2]
    ratio = min(target_h / h0, target_w / w0)
    new_h, new_w = int(round(h0 * ratio)), int(round(w0 * ratio))

    # Resize with bilinear interpolation (cv2 not required – use numpy/PIL)
    try:
        import cv2  # noqa: PLC0415
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image  # noqa: PLC0415
        pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        resized = np.array(pil)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, ratio


def _nms(
    boxes: "np.ndarray",
    scores: "np.ndarray",
    iou_thr: float,
) -> list[int]:
    """
    Pure-NumPy greedy NMS.  Returns indices of kept boxes sorted by score.

    Args:
        boxes   : (N, 4) array in xyxy format.
        scores  : (N,) confidence scores.
        iou_thr : IoU threshold above which a box is suppressed.
    """
    import numpy as np  # noqa: PLC0415

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thr]

    return keep


def _build_stride_scale(
    num_anchors: int,
    stride_splits: "list[tuple[int, int]] | None" = None,
) -> "np.ndarray":
    """
    Build a per-anchor stride scale vector for YOLOX FPN outputs.

    YOLOX concatenates predictions from three feature-map heads in order
    (small → medium → large stride).  The raw ``cx, cy, w, h`` values from
    this ONNX model are expressed in **grid-cell units** and must be multiplied
    by the corresponding stride before any further coordinate transformation.

    Args:
        num_anchors   : Total number of anchors in the output tensor (axis 1).
        stride_splits : List of ``(count, stride)`` tuples that partition the
                        anchor axis.  Defaults to
                        :data:`YOLOX_STRIDE_SPLITS` (2704/676/169 for a
                        416×416 input).

    Returns:
        ``(num_anchors,)`` float32 array where each element is the stride that
        applies to the corresponding anchor.
    """
    import numpy as np  # noqa: PLC0415

    if stride_splits is None:
        stride_splits = YOLOX_STRIDE_SPLITS

    total = sum(c for c, _ in stride_splits)
    if total != num_anchors:
        raise ValueError(
            f"[_build_stride_scale] stride_splits sum ({total}) does not match "
            f"num_anchors ({num_anchors}).  Pass the correct stride_splits for "
            "this model."
        )

    scale = np.empty(num_anchors, dtype=np.float32)
    offset = 0
    for count, stride in stride_splits:
        scale[offset: offset + count] = stride
        offset += count
    return scale


def _decode_bboxes(
    boxes_input: "np.ndarray",
    ratio: float,
    orig_h: int,
    orig_w: int,
    stride_scale: "np.ndarray | None" = None,
) -> "np.ndarray":
    """
    Decode bounding boxes from YOLOX raw output to original-image pixel domain.

    YOLOX ONNX outputs ``cx, cy, w, h`` coordinates in **grid-cell units**
    (i.e. the values must first be multiplied by the FPN stride to obtain
    input-image pixel coordinates).  This function performs the full decode:

      1. Accepts boxes in ``x1, y1, x2, y2`` (xyxy) format computed from the
         raw ``cx, cy, w, h`` predictions (still in grid-cell units).
      2. **Applies per-anchor stride scaling** (×8 / ×16 / ×32) to convert
         from grid-cell units to input-image pixel space.
      3. Divides each coordinate by the letterbox ``ratio`` to undo the
         letterbox scaling and bring values to original-image pixel domain.
      4. Clamps every coordinate to the valid image boundary
         ``[0, orig_w]`` (x-axis) / ``[0, orig_h]`` (y-axis).

    Args:
        boxes_input  : ``(N, 4)`` float array of xyxy boxes in grid-cell units.
        ratio        : Letterbox scale factor returned by :func:`_letterbox`
                       (``min(input_h / orig_h, input_w / orig_w)``).
        orig_h       : Height of the original image in pixels.
        orig_w       : Width  of the original image in pixels.
        stride_scale : ``(N,)`` per-anchor stride array produced by
                       :func:`_build_stride_scale`.  When ``None`` the function
                       skips stride scaling (use this only when boxes are
                       already in input-image pixel space).

    Returns:
        ``(N, 4)`` float32 array of xyxy boxes in original-image pixel space,
        clamped to ``[0, orig_w] × [0, orig_h]``.
    """
    import numpy as np  # noqa: PLC0415

    decoded = boxes_input.copy().astype(np.float32)

    # ── Step 1: grid-cell units → input-image pixel space ─────────────────
    # Multiply each box coordinate by its anchor's FPN stride.
    # stride_scale shape: (N,) → broadcast to (N, 4) via [:, None].
    if stride_scale is not None:
        decoded *= stride_scale[:, None]

    # ── Step 2: undo letterbox scaling → original-image pixel space ───────
    decoded /= ratio

    # ── Step 3: clamp to image boundaries ─────────────────────────────────
    decoded[:, 0] = np.clip(decoded[:, 0], 0.0, orig_w)   # x1
    decoded[:, 2] = np.clip(decoded[:, 2], 0.0, orig_w)   # x2
    decoded[:, 1] = np.clip(decoded[:, 1], 0.0, orig_h)   # y1
    decoded[:, 3] = np.clip(decoded[:, 3], 0.0, orig_h)   # y2

    return decoded


def _postprocess_onnx_output(
    raw: "np.ndarray",
    input_h: int,
    input_w: int,
    orig_h: int,
    orig_w: int,
    ratio: float,
    conf_thr: float = 0.01,
    nms_thr:  float = 0.65,
    num_classes: int = 80,
) -> list[dict]:
    """
    Convert the raw ONNX output tensor to a list of COCO-format detections.

    YOLOX ONNX output shape: (1, num_anchors, 5 + num_classes)
      - columns 0-3 : cx, cy, w, h  (in grid-cell units; must be multiplied by
                      the FPN stride to reach input-image pixel space)
      - column  4   : objectness score
      - columns 5+  : per-class scores

    Args:
        raw         : numpy array of shape (1, A, 5+C).
        input_h/w   : spatial dimensions of the model input (after letterbox).
        orig_h/w    : original image dimensions (before letterbox).
        ratio       : letterbox scale factor (output of _letterbox).
        conf_thr    : minimum objectness × class-score to keep a detection.
        nms_thr     : IoU threshold for NMS.
        num_classes : number of object classes (80 for COCO).

    Returns:
        List of dicts with keys: bbox (xywh, original scale), score, class_idx.
    """
    import numpy as np  # noqa: PLC0415

    pred = raw[0]  # (A, 5+C)
    num_anchors = pred.shape[0]

    # ── Build per-anchor stride scale (×8 / ×16 / ×32) ───────────────────
    stride_scale = _build_stride_scale(num_anchors)   # (A,)

    # ── Convert cx,cy,w,h → x1,y1,x2,y2 (still in grid-cell units) ───────
    cx, cy, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = cx - pw / 2.0
    y1 = cy - ph / 2.0
    x2 = cx + pw / 2.0
    y2 = cy + ph / 2.0
    boxes_input = np.stack([x1, y1, x2, y2], axis=1)  # (A, 4)

    obj_scores   = pred[:, 4]                          # (A,)
    class_scores = pred[:, 5: 5 + num_classes]         # (A, C)

    # ── Per-class confidence = objectness × class probability ─────────────
    scores_all = obj_scores[:, None] * class_scores    # (A, C)
    class_ids  = np.argmax(scores_all, axis=1)         # (A,)
    max_scores = scores_all[np.arange(len(class_ids)), class_ids]  # (A,)

    # ── Confidence filter ──────────────────────────────────────────────────
    mask = max_scores >= conf_thr
    if not mask.any():
        return []

    boxes_f        = boxes_input[mask]
    scores_f       = max_scores[mask]
    cls_f          = class_ids[mask]
    stride_scale_f = stride_scale[mask]   # keep stride aligned with filtered boxes

    # ── Per-class NMS ──────────────────────────────────────────────────────
    results: list[dict] = []
    for cls_idx in np.unique(cls_f):
        sel  = cls_f == cls_idx
        kept = _nms(boxes_f[sel], scores_f[sel], nms_thr)

        # Decode the surviving boxes:
        #   grid-cell units  ──×stride──▶  input-image pixels
        #                    ──÷ratio───▶  original-image pixels
        #                    ──clamp────▶  within image boundary
        decoded = _decode_bboxes(
            boxes_f[sel][kept],
            ratio, orig_h, orig_w,
            stride_scale=stride_scale_f[sel][kept],
        )

        for i, k in enumerate(kept):
            bx1, by1, bx2, by2 = decoded[i]
            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0 or bh <= 0:
                continue
            results.append({
                "bbox":      [float(bx1), float(by1), float(bw), float(bh)],
                "score":     float(scores_f[sel][k]),
                "class_idx": int(cls_idx),
            })

    return results


def verify_onnx_with_coco(
    onnx_path: str,
    coco_dir: str,
    input_size: tuple[int, int],
    num_images: int = 500,
    conf_thr: float = 0.01,
    nms_thr:  float = 0.65,
) -> None:
    """
    Evaluate the exported ONNX model on a subset of COCO val2017 and report
    mAP@[0.50:0.95] and mAP@0.50 using pycocotools.

    The function:
      1. Ensures onnxruntime, numpy, and pycocotools are available.
      2. Downloads COCO val2017 images + annotations if not already present.
      3. Runs the ONNX model on up to *num_images* validation images.
      4. Converts predictions to COCO JSON format and calls COCOeval.

    Args:
        onnx_path  : Path to the ONNX model to evaluate.
        coco_dir   : Directory where COCO data will be stored / is already stored.
                     Expected layout after download:
                       <coco_dir>/val2017/          ← JPEG images
                       <coco_dir>/annotations/
                           instances_val2017.json   ← ground-truth annotations
        input_size : (height, width) fed to the model.
        num_images : Maximum number of val images to evaluate (default 500).
                     Pass 0 or a negative value to evaluate the full 5 000-image
                     val2017 set (slow – ~30 min on CPU).
        conf_thr   : Objectness × class-score threshold for keeping detections.
        nms_thr    : IoU threshold used in per-class NMS.
    """

    # ── Step V-A: ensure runtime dependencies ─────────────────────────────
    print("[VERIFY] Checking verification dependencies …")
    ensure_dependencies({
        "onnxruntime": "onnxruntime",
        "numpy":       "numpy",
    })
    # pycocotools ships as 'pycocotools' on PyPI but imports as 'pycocotools'
    try:
        importlib.import_module("pycocotools")
        print("[DEP]  ✔  pycocotools is already installed.")
    except ImportError:
        print("[DEP]  ✘  pycocotools not found – installing …")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pycocotools"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[VERIFY] ERROR: could not install pycocotools "
                f"(exit code {exc.returncode}).\n"
                "         Please install it manually: pip install pycocotools"
            )
            return

    import numpy as np                          # noqa: PLC0415
    import onnxruntime as ort                   # noqa: PLC0415
    from pycocotools.coco import COCO           # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval   # noqa: PLC0415

    # ── Step V-B: prepare COCO data directories ───────────────────────────
    images_dir  = os.path.join(coco_dir, "val2017")
    annots_dir  = os.path.join(coco_dir, "annotations")
    annots_file = os.path.join(annots_dir, "instances_val2017.json")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annots_dir, exist_ok=True)

    # ── Step V-C: download annotations if missing ─────────────────────────
    if not os.path.exists(annots_file):
        print("[VERIFY] Annotations not found – downloading …")
        annots_zip = os.path.join(coco_dir, "annotations_trainval2017.zip")
        ok = download_file(COCO_VAL_ANNOTS_URL, annots_zip, label="COCO annotations")
        if not ok:
            print("[VERIFY] ERROR: could not download COCO annotations. Skipping verification.")
            return
        print("[VERIFY] Extracting annotations …")
        import zipfile  # noqa: PLC0415
        with zipfile.ZipFile(annots_zip, "r") as zf:
            zf.extractall(coco_dir)
        os.remove(annots_zip)

    # ── Step V-D: load COCO ground-truth ──────────────────────────────────
    print(f"[VERIFY] Loading COCO ground-truth from: {annots_file}")
    # Suppress pycocotools' verbose stdout during loading
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(annots_file)

    all_img_ids: list[int] = sorted(coco_gt.getImgIds())
    if num_images > 0:
        eval_img_ids = all_img_ids[:num_images]
    else:
        eval_img_ids = all_img_ids

    print(
        f"[VERIFY] Will evaluate on {len(eval_img_ids)} / {len(all_img_ids)} "
        "val2017 images."
    )

    # ── Step V-E: download images if the directory is empty ───────────────
    # Check whether the first image in our eval set is already on disk.
    first_info = coco_gt.loadImgs(eval_img_ids[0])[0]
    first_path = os.path.join(images_dir, first_info["file_name"])
    if not os.path.exists(first_path):
        print("[VERIFY] val2017 images not found – downloading (~1 GB) …")
        images_zip = os.path.join(coco_dir, "val2017.zip")
        ok = download_file(COCO_VAL_IMAGES_URL, images_zip, label="COCO val2017 images")
        if not ok:
            print("[VERIFY] ERROR: could not download COCO images. Skipping verification.")
            return
        print("[VERIFY] Extracting images …")
        import zipfile  # noqa: PLC0415
        with zipfile.ZipFile(images_zip, "r") as zf:
            zf.extractall(coco_dir)
        os.remove(images_zip)

    # ── Step V-F: create ONNX Runtime session ─────────────────────────────
    print(f"[VERIFY] Loading ONNX model: {onnx_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3   # suppress ORT verbose logs
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name  = session.get_inputs()[0].name
    input_h, input_w = input_size

    print(
        f"[VERIFY] Running inference "
        f"(input {input_h}×{input_w}, conf≥{conf_thr}, NMS IoU≤{nms_thr}) …"
    )

    # ── Step V-G: run inference and collect predictions ────────────────────
    coco_predictions: list[dict] = []
    skipped = 0

    for idx, img_id in enumerate(eval_img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Load image (try cv2 first, fall back to PIL)
        try:
            import cv2  # noqa: PLC0415
            bgr = cv2.imread(img_path)
            if bgr is None:
                skipped += 1
                continue
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image  # noqa: PLC0415
            pil_img = Image.open(img_path).convert("RGB")
            img_rgb = np.array(pil_img)

        orig_h, orig_w = img_rgb.shape[:2]

        # Letterbox resize to model input size
        padded, ratio = _letterbox(img_rgb, input_h, input_w)

        # Pre-process: mean=0, scale=255  →  output = (pixel - 0) / 255.0
        inp = (padded.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]  # (1, 3, H, W)

        # ONNX inference
        raw_out = session.run(None, {input_name: inp})  # list of arrays
        raw     = raw_out[0]                            # (1, A, 5+C)

        # Postprocess
        dets = _postprocess_onnx_output(
            raw, input_h, input_w, orig_h, orig_w, ratio,
            conf_thr=conf_thr, nms_thr=nms_thr,
        )

        for det in dets:
            coco_predictions.append({
                "image_id":   img_id,
                "category_id": COCO80_CATEGORY_IDS[det["class_idx"]],
                "bbox":        det["bbox"],   # [x, y, w, h] in original-image pixels
                "score":       det["score"],
            })

        # Progress every 50 images
        if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_img_ids):
            sys.stdout.write(
                f"\r[VERIFY] {idx + 1}/{len(eval_img_ids)} images processed "
                f"({len(coco_predictions)} detections so far) …"
            )
            sys.stdout.flush()

    print()  # newline after progress line

    if skipped:
        print(f"[VERIFY] Warning: {skipped} image(s) were skipped (file not found).")

    # ── Step V-H: run COCOeval ─────────────────────────────────────────────
    if not coco_predictions:
        print("[VERIFY] No detections produced – cannot compute mAP.")
        return

    print(f"[VERIFY] Total detections: {len(coco_predictions)}")
    print("[VERIFY] Running COCOeval …")

    # Write predictions to a temp file (pycocotools requires a file path or list)
    _, tmp_pred_path = tempfile.mkstemp(suffix=".json")
    try:
        with open(tmp_pred_path, "w") as fh:
            json.dump(coco_predictions, fh)

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(tmp_pred_path)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = eval_img_ids   # restrict to evaluated images
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        os.remove(tmp_pred_path)

    # Capture and print the summary table
    summary_buf = io.StringIO()
    with contextlib.redirect_stdout(summary_buf):
        coco_eval.summarize()
    summary_str = summary_buf.getvalue()

    ap50_95 = float(coco_eval.stats[0])
    ap50    = float(coco_eval.stats[1])

    print("\n" + "=" * 60)
    print("  COCO val2017 Accuracy Verification Results")
    print("=" * 60)
    print(summary_str)
    print(f"  mAP@[0.50:0.95] : {ap50_95:.4f}  ({ap50_95 * 100:.2f} %)")
    print(f"  mAP@0.50        : {ap50:.4f}  ({ap50 * 100:.2f} %)")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# Helper function for ONNX simplification
# ─────────────────────────────────────────────

def handle_simplification(input_path: str, output_path: str, use_temp_file: bool, args, model_name: str) -> str:
    """
    Attempt to simplify the ONNX model at input_path and save to output_path.
    Handles temp file cleanup/move and returns the path to the model that should be used for verification.
    """
    try:
        import onnx
        import onnxsim

        print(f"[SIMPLIFY] Simplifying model: {input_path}")
        print(f"[SIMPLIFY] Output will be saved to: {output_path}")

        model = onnx.load(input_path)
        model_simplified, check = onnxsim.simplify(
            model,
            check_n=3,
            perform_optimization=True,
            skip_fuse_bn=False,
        )

        if not check:
            print("[SIMPLIFY] Warning: Simplification validation failed")
            print("         The simplified model may not produce identical outputs")
            print("         Proceeding anyway, but please verify the model manually")
        else:
            print("[SIMPLIFY] Simplification successful and validated")

        onnx.save(model_simplified, output_path)
        print(f"[SIMPLIFY] Simplified model saved to: {output_path}\n")

        # If we used a temporary file for input, remove it now
        if use_temp_file and input_path != output_path:
            os.remove(input_path)

        return output_path   # success: use the simplified model at output_path

    except ImportError:
        print(
            "[SIMPLIFY] WARNING: 'onnx-simplifier' package not found – skipping simplification.\n"
            "         Install it with:  pip install onnx-simplifier\n"
        )
        # If we were using a temp file, move it to the final location
        if use_temp_file and input_path != output_path:
            shutil.move(input_path, output_path)
            print(f"[INFO] Moved model to: {output_path}")
        return output_path

    except Exception as exc:
        print(f"[SIMPLIFY] WARNING: simplification failed ({exc}) – using original model.\n")
        # If we were using a temp file, move it to the final location
        if use_temp_file and input_path != output_path:
            shutil.move(input_path, output_path)
            print(f"[INFO] Moved model to: {output_path}")
        return output_path


# ─────────────────────────────────────────────
# Main – all configuration lives here
# ─────────────────────────────────────────────

def main() -> None:
    """
    Entry-point logic.

    All configuration variables are defined here so they are easy to find
    and modify without touching the helper functions above.

    Download strategy (for each model):
      1. Attempt to download the pre-built ONNX from GitHub releases.
      2. If that fails, download the .pth checkpoint and convert it to ONNX.
    """

    # ── CLI argument parsing ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Download (or build) YOLOX ONNX model(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=[MODEL_NAME],
        choices=list(MODEL_EXP_NAME.keys()),
        help="YOLOX variant(s) to download. Can specify multiple models.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help=(
            "After obtaining the ONNX model, verify its accuracy on COCO val2017. "
            "Images and annotations are downloaded automatically if not present."
        ),
    )
    parser.add_argument(
        "--coco-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_coco_data"),
        metavar="DIR",
        help="Directory where COCO val2017 data is stored (or will be downloaded to).",
    )
    parser.add_argument(
        "--num-val-images",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Number of COCO val2017 images to use for verification. "
            "Use 0 to evaluate the full 5 000-image set (slow on CPU)."
        ),
    )
    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.01,
        metavar="T",
        help="Objectness × class-score threshold for keeping detections during verification.",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=0.65,
        metavar="T",
        help="IoU threshold used in per-class NMS during verification.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify the ONNX model using onnx-simplifier after download/conversion.",
    )
    args = parser.parse_args()

    # Whether we need a temporary file for intermediate processing (when simplifying)
    use_temp_file = args.simplify

    # Destination directory – saves alongside this script by default
    save_dir = os.path.dirname(os.path.abspath(__file__))

    # Check / install base dependencies once (only needed for simplification)
    ensure_dependencies(REQUIRED_PACKAGES)

    # Process each model
    for model_name in args.model:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}\n")

        # ── Configuration for this model ────────────────────────────────────

        # ── Read URLs from .link files ──────────────────────────────────────
        # Each model variant has two .link files next to this script:
        #   <model_name>.onnx.link  – pre-built ONNX (primary source)
        #   <model_name>.pth.link   – PyTorch checkpoint (fallback source)
        onnx_link_path = os.path.join(save_dir, f"{model_name}.onnx.link")
        pth_link_path  = os.path.join(save_dir, f"{model_name}.pth.link")

        try:
            onnx_url = read_url_from_link_file(onnx_link_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] Could not read ONNX URL from link file for {model_name}: {exc}")
            print("[ERROR] Skipping this model and continuing with others...")
            continue

        try:
            pth_url = read_url_from_link_file(pth_link_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] Could not read PTH URL from link file for {model_name}: {exc}")
            print("[ERROR] Skipping this model and continuing with others...")
            continue

        # Output filenames
        onnx_filename = f"{model_name}.onnx"
        pth_filename  = f"{model_name}.pth"

        # Full destination paths
        onnx_path = os.path.join(save_dir, onnx_filename)
        pth_path  = os.path.join(save_dir, pth_filename)

        # Input resolution for this variant (height, width)
        input_size: tuple[int, int] = MODEL_INPUT_SIZE.get(model_name, (640, 640))

        print(f"[INFO] Target model : {model_name}")
        print(f"[INFO] Input size   : {input_size[0]}×{input_size[1]}")
        print()

        # ── Step 2: Try to download the pre-built ONNX ─────────────────────
        print("=" * 60)
        print(f"  Strategy 1 – Download pre-built ONNX for {model_name}")
        print("=" * 60)

        # Determine if we need to use a temporary file for processing
        if use_temp_file:
            # Create a temporary file for intermediate processing
            temp_fd, temp_path = tempfile.mkstemp(suffix='.onnx', prefix=f'{model_name}_')
            os.close(temp_fd)
            os.remove(temp_path)  # mkstemp creates an empty placeholder; remove it so download_file won't skip the download
            download_path = temp_path
        else:
            download_path = onnx_path

        onnx_ok = download_file(onnx_url, download_path, label=f"{model_name} ONNX")

        if onnx_ok:
            # ── Post-process: run shape inference on the downloaded ONNX ──────────
            # Pre-built ONNX files from GitHub releases may have symbolic or
            # incomplete shape annotations.  Running onnx.shape_inference ensures
            # that all intermediate tensors carry correct shape information, which
            # is required by many downstream tools (e.g. TFLite converters, TVM,
            # TIDL, onnxsim).
            print("=" * 60)
            print("  Post-processing – ONNX shape inference")
            print("=" * 60)
            try:
                import onnx                  # noqa: PLC0415
                import onnx.shape_inference  # noqa: PLC0415

                print(f"[POST] Running ONNX shape inference on: {download_path}")
                model_proto = onnx.load(download_path)
                model_proto = onnx.shape_inference.infer_shapes(model_proto)
                onnx.save(model_proto, download_path)
                print("[POST] Shape inference complete – model saved.\n")
            except ImportError:
                print(
                    "[POST] WARNING: 'onnx' package not found – skipping shape inference.\n"
                    "       Install it with:  pip install onnx\n"
                )
            except Exception as exc:
                print(f"[POST] WARNING: shape inference failed ({exc}) – model unchanged.\n")

            # Optional ONNX simplification
            final_model_path = handle_simplification(
                download_path,
                onnx_path,
                use_temp_file,
                args,
                model_name,
            )

            # Primary path succeeded – proceed to optional verification
            if args.verify:
                verify_onnx_with_coco(
                    final_model_path,
                    coco_dir=args.coco_dir,
                    input_size=input_size,
                    num_images=args.num_val_images,
                    conf_thr=args.conf_thr,
                    nms_thr=args.nms_thr,
                )
            continue  # Move to next model

        # ── Step 3: Fallback – download .pth and convert to ONNX ───────────────
        print("=" * 60)
        print(f"  Strategy 2 – Download .pth checkpoint and convert to ONNX for {model_name}")
        print("=" * 60)

        pth_ok = download_file(pth_url, pth_path, label=f"{model_name} PTH checkpoint")

        if not pth_ok:
            print("[ERROR] Both download strategies failed.")
            print("        Please check your internet connection and try again.")
            print("[ERROR] Skipping this model and continuing with others...")
            continue

        # Convert the downloaded .pth to .onnx
        if use_temp_file:
            # Create a temporary file for intermediate processing
            temp_fd, temp_path = tempfile.mkstemp(suffix='.onnx', prefix=f'{model_name}_')
            os.close(temp_fd)
            os.remove(temp_path)  # mkstemp creates an empty placeholder; remove it so download_file won't skip the download
            convert_path = temp_path
        else:
            convert_path = onnx_path

        convert_pth_to_onnx(pth_path, convert_path, model_name=model_name, input_size=input_size)

        # Optional ONNX simplification
        final_model_path = handle_simplification(
            convert_path,
            onnx_path,
            use_temp_file,
            args,
            model_name,
        )

        if args.verify:
            verify_onnx_with_coco(
                final_model_path,
                coco_dir=args.coco_dir,
                input_size=input_size,
                num_images=args.num_val_images,
                conf_thr=args.conf_thr,
                nms_thr=args.nms_thr,
            )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()