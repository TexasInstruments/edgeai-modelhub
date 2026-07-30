# Copyright (c) 2018-2026, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# ---------------------------------------------------------------------------
# Script: prepare_model.py
#
# Purpose:
#   Export one or more YOLOv8 model variants (n / s / m / l / x) to ONNX
#   (or any other Ultralytics-supported format) using the Ultralytics Python
#   API.  Required packages (onnx, ultralytics) are installed automatically
#   at runtime if they are not already present.
#
# Usage examples:
#   # Export only yolov8n (default behaviour)
#   python prepare_model.py
#
#   # Export specific variants
#   python prepare_model.py --models yolov8n yolov8s
#
#   # Export all variants
#   python prepare_model.py --models yolov8n yolov8s yolov8m yolov8l yolov8x
#
#   # Export to a custom directory in TorchScript format
#   python prepare_model.py --models yolov8n --format torchscript --output-dir ./exports
# ---------------------------------------------------------------------------

import argparse
from pathlib import Path
import sys
import subprocess


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def install_package(package: str) -> None:
    """Install a Python package at runtime using pip.

    This ensures that optional heavyweight dependencies (e.g. ``onnx``,
    ``ultralytics``) are available without requiring them to be listed in the
    project's top-level requirements file.

    Args:
        package (str): Package name as accepted by ``pip install`` (e.g.
            ``'onnx'``, ``'ultralytics>=8.0'``).
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def export_model(checkpoint: str, output_format: str = 'onnx', output_dir: str = None, opset: int = None) -> str:
    """Export a YOLOv8 model checkpoint to the requested format.

    The function loads the model via the Ultralytics ``YOLO`` class and calls
    its ``export()`` method.  When *checkpoint* is a bare filename such as
    ``'yolov8n.pt'``, Ultralytics will automatically download the pre-trained
    weights from the official release page on first use.

    Supported export formats (non-exhaustive):
        ``onnx``, ``torchscript``, ``tflite``, ``pb``, ``saved_model``,
        ``coreml``, ``paddle``, ``ncnn``.

    Args:
        checkpoint (str): Path to, or name of, the YOLO checkpoint file
            (e.g. ``'yolov8n.pt'`` or ``'/path/to/custom_model.pt'``).
        output_format (str): Target export format understood by Ultralytics.
            Defaults to ``'onnx'``.
        output_dir (str | None): Directory where the exported artefact will be
            written.  The directory is created automatically if it does not
            exist.  When ``None`` the Ultralytics default location (same
            folder as the source ``.pt`` file) is used.
        opset (int | None): ONNX opset version to use for export. Only applies
            when output_format is 'onnx'. When ``None``, uses Ultralytics default.

    Returns:
        str: Absolute path to the exported model file as reported by
        Ultralytics.
    """
    # Import here so that the module can be imported without ultralytics
    # being installed (the install happens in __main__ before any export call).
    from ultralytics.models import YOLO

    print(f"\n[export] Loading checkpoint: {checkpoint}")
    model = YOLO(checkpoint)

    # Build keyword arguments for model.export()
    export_kwargs = {
        'format': output_format,
    }

    # import torch
    # dummy_input = torch.zeros(3, 640, 640, dtype=torch.float32)
    # torch.onnx.export(
    #     model=
    # )

    if output_dir:
        # Resolve and create the output directory tree as needed
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_kwargs['project'] = str(output_dir)
    
    if opset is not None and output_format == 'onnx':
        export_kwargs['opset'] = opset

    print(f"[export] Exporting to format='{output_format}'"
          + (f", output_dir='{output_dir}'" if output_dir else "") + " …")

    exported_path = model.export(**export_kwargs)
    print(f"[export] Model exported successfully to: {exported_path}")
    return exported_path


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

# All recognised YOLOv8 size variants in ascending order of complexity
ALL_MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

# Default: export only the nano variant (smallest / fastest)
DEFAULT_MODELS = ['yolov8n']


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the export script.

    Returns:
        argparse.Namespace: Parsed argument object with the following fields:

        * ``models``     – list of model base-names to export (without ``.pt``)
        * ``format``     – Ultralytics export format string
        * ``output_dir`` – destination directory (``None`` → Ultralytics default)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLOv8 model variants to ONNX (or another format) "
            "using the Ultralytics Python API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=ALL_MODELS,
        default=DEFAULT_MODELS,
        metavar='MODEL',
        help=(
            "One or more YOLOv8 variants to export.  "
            f"Choices: {ALL_MODELS}.  "
            f"Default: {DEFAULT_MODELS}."
        ),
    )

    parser.add_argument(
        '--format',
        default='onnx',
        metavar='FORMAT',
        help=(
            "Ultralytics export format.  "
            "Common values: onnx, torchscript, tflite, pb, saved_model, coreml."
        ),
    )

    parser.add_argument(
        '--output-dir',
        default=None,
        metavar='DIR',
        help=(
            "Directory to save exported models.  "
            "Created automatically if it does not exist.  "
            "Defaults to the Ultralytics standard location when omitted."
        ),
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=19,
        metavar='VERSION',
        help=(
            "ONNX opset version to use for export (e.g., 19).  "
            "Only applies when --format is 'onnx'.  "
            "Defaults to Ultralytics default opset when omitted."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 1. Ensure runtime dependencies are available
    # ------------------------------------------------------------------
    # 'onnx' is required for ONNX export; 'ultralytics' provides the YOLO API.
    install_package('onnx')
    install_package('ultralytics')

    # ------------------------------------------------------------------
    # 2. Parse CLI arguments
    # ------------------------------------------------------------------
    args = parse_args()

    print(f"Models to export : {args.models}")
    print(f"Export format    : {args.format}")
    print(f"Output directory : {args.output_dir or '(Ultralytics default)'}")

    # ------------------------------------------------------------------
    # 3. Export each requested model variant
    # ------------------------------------------------------------------
    for model_name in args.models:
        # Append the '.pt' extension expected by Ultralytics
        checkpoint = f"{model_name}.pt"
        export_model(
            checkpoint=checkpoint,
            output_format=args.format,
            output_dir=args.output_dir,
            opset=args.opset,
        )

    print("\n[done] All requested models exported.")
