#!/usr/bin/env python3

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

"""
Unified script to download ONNX models and fix their shapes for hardware deployment.

Reads .link files containing download URLs and automatically:
1. Downloads the model if not present
2. Fixes dynamic shapes to static shapes
3. Validates the result

Link file format:
    <download_url> -o <output_filename>

Example:
    https://huggingface.co/.../model.onnx -o yolo26n.onnx
"""

import sys
import subprocess
from pathlib import Path
import argparse


def _ensure_dependencies():
    required = {"onnx": "onnx", "onnxsim": "onnx-simplifier"}
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing missing dependency: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Supported YOLO26 model variants
YOLO26_VARIANTS = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']
DEFAULT_MODEL = YOLO26_VARIANTS[0]


def list_models(script_dir):
    """
    Print all supported YOLO26 model variants along with their local status:
    whether a .link file is present and whether the final ONNX output exists.
    """
    col = 14
    header = f"  {'Variant':<{col}}{'Link file':<12}{'ONNX output':<30}{'Status'}"
    print("\n" + "=" * len(header))
    print("  Available YOLO26 model variants")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for variant in YOLO26_VARIANTS:
        link_file = script_dir / f'{variant}.onnx.link'

        if not link_file.exists():
            print(f"  {variant:<{col}}{'missing':<12}{'-':<30}{'no link file'}")
            continue

        _, model_filename = parse_link_file(link_file)
        model_filename = model_filename or '?'
        final_output = script_dir / model_filename

        if final_output.exists():
            file_size = final_output.stat().st_size
            status = f"downloaded ({file_size / 1024 / 1024:.1f} MB)"
        else:
            status = "not downloaded"

        print(f"  {variant:<{col}}{'ok':<12}{model_filename:<30}{status}")

    print("=" * len(header) + "\n")


def parse_link_file(link_file_path):
    """
    Parse .link file to extract download URL and output filename.

    Expected format: <URL> -o <filename>

    Returns:
        tuple: (download_url, output_filename) or (None, None) if parsing fails
    """
    try:
        with open(link_file_path, 'r') as f:
            content = f.read().strip()

        # Find the line with the URL and -o flag
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Look for pattern: URL -o filename
            if '-o' in line:
                parts = line.split('-o')
                if len(parts) == 2:
                    url = parts[0].strip()
                    filename = parts[1].strip()

                    # Convert HuggingFace blob URLs to resolve URLs for direct download
                    if 'huggingface.co' in url and '/blob/' in url:
                        url = url.replace('/blob/', '/resolve/')

                    return url, filename

        print(f"Error: Could not parse link file format. Expected: <URL> -o <filename>")
        return None, None

    except Exception as e:
        print(f"Error reading link file: {e}")
        return None, None


def download_model(url, output_path, force=False):
    """
    Download model from URL using curl.

    Args:
        url: Download URL
        output_path: Path to save downloaded model
        force: Force re-download even if file exists

    Returns:
        bool: True if successful, False otherwise
    """
    if output_path.exists() and not force:
        print(f"Model already exists: {output_path}")
        file_size = output_path.stat().st_size
        print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        return True

    print(f"\n📥 Downloading Model:")
    print("-" * 80)
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print()

    try:
        # Use curl to download with progress bar
        result = subprocess.run(
            ['curl', '-L', url, '-o', str(output_path), '--progress-bar'],
            check=True,
            capture_output=False
        )

        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"\n✓ Download completed successfully!")
            print(f"✓ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            return True
        else:
            print(f"\n✗ Download failed: output file not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ curl not found. Please install curl.")
        return False


def fix_model_shape(model_path, output_path, batch_size=1, channels=3, height=224, width=224, use_simplifier=True):
    """
    Convert dynamic ONNX model input shape to fixed shape in all layers.

    Uses ONNX shape inference to propagate fixed shapes through all intermediate layers.
    Optionally uses onnxsim for additional simplification and optimization.
    """
    import onnx
    from onnx import shape_inference

    print(f"\n🔧 Fixing Model Shapes:")
    print("=" * 80)
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")

    # Load model
    print(f"\nLoading model...")
    model = onnx.load(str(model_path))

    # Get the first input (skip initializers)
    graph = model.graph
    input_tensor = None
    for inp in graph.input:
        if any(init.name == inp.name for init in graph.initializer):
            continue
        input_tensor = inp
        break

    if input_tensor is None:
        print("✗ Error: No input tensor found!")
        return False

    # Print original shape
    print(f"\n📋 Original Shape:")
    print("-" * 80)
    print(f"Input name: {input_tensor.name}")
    original_shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            original_shape.append(str(dim.dim_value))
        elif dim.dim_param:
            original_shape.append(f"'{dim.dim_param}'")
        else:
            original_shape.append("?")
    print(f"Shape: [{', '.join(original_shape)}]")

    # Modify input shape to fixed dimensions
    print(f"\n🔧 Setting Fixed Shape:")
    print("-" * 80)
    new_shape = [batch_size, channels, height, width]
    print(f"New shape: {new_shape}")
    print(f"Format: [batch_size, channels, height, width]")

    # Clear existing dimensions and add new fixed dimensions
    input_tensor.type.tensor_type.shape.ClearField('dim')
    for dim_value in new_shape:
        dim = input_tensor.type.tensor_type.shape.dim.add()
        dim.dim_value = dim_value

    # Run shape inference
    print(f"\n🔄 Running Shape Inference:")
    print("-" * 80)
    try:
        model = shape_inference.infer_shapes(model)
        value_info_count = len(model.graph.value_info)
        print(f"✓ Propagated shapes through {value_info_count} intermediate tensors")
    except Exception as e:
        print(f"⚠ Warning: Shape inference issue: {e}")
        print("  Continuing with partial inference...")

    # Validate model
    print(f"\n✅ Validating Model:")
    print("-" * 80)
    try:
        onnx.checker.check_model(model)
        print("✓ Model validation passed")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False

    # Optional: Use onnx-simplifier
    if use_simplifier:
        print(f"\n🚀 Running ONNX Simplifier:")
        print("-" * 80)
        try:
            import onnxsim
            model_simplified, check = onnxsim.simplify(
                model,
                check_n=3,
                perform_optimization=True,
                skip_fuse_bn=False,
                overwrite_input_shapes={input_tensor.name: new_shape}
            )

            if check:
                print("✓ Model simplified and optimized")
                model = model_simplified

                # Report node reduction if any
                original_nodes = len(graph.node)
                simplified_nodes = len(model.graph.node)
                if simplified_nodes < original_nodes:
                    print(f"✓ Reduced nodes: {original_nodes} → {simplified_nodes}")
            else:
                print("⚠ Simplification validation failed, using non-simplified version")

        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping")
            print("  Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"⚠ Simplification failed: {e}")
            print("  Continuing with non-simplified model")

    # Save the modified model
    print(f"\n💾 Saving Fixed Model:")
    print("-" * 80)
    onnx.save(model, str(output_path))
    output_size = output_path.stat().st_size
    print(f"✓ Saved to: {output_path}")
    print(f"✓ File size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")

    # Final verification
    print(f"\n🔍 Final Verification:")
    print("-" * 80)
    try:
        verified_model = onnx.load(str(output_path))
        onnx.checker.check_model(verified_model)

        # Check input shape
        verified_graph = verified_model.graph
        for inp in verified_graph.input:
            if any(init.name == inp.name for init in verified_graph.initializer):
                continue
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            all_fixed = all(isinstance(s, int) and s > 0 for s in shape)
            if all_fixed:
                print(f"✓ Input '{inp.name}': {shape}")
            else:
                print(f"⚠ Input '{inp.name}' has dynamic dimensions")

        # Check intermediate tensors
        if verified_graph.value_info:
            fixed_count = sum(
                1 for vi in verified_graph.value_info
                if all(dim.dim_value > 0 for dim in vi.type.tensor_type.shape.dim)
            )
            total_count = len(verified_graph.value_info)
            print(f"✓ Fixed shapes: {fixed_count}/{total_count} intermediate tensors")

        print(f"\n✨ Success! Fixed model ready for deployment")
        return True

    except Exception as e:
        print(f"✗ Final verification failed: {e}")
        return False


def process_single_model(model_variant, script_dir, args):
    """
    Process a single model variant.
    
    Args:
        model_variant: Name of the model variant (e.g., 'yolo26n')
        script_dir: Directory containing the script and link files
        args: Parsed command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    link_file_name = f'{model_variant}.onnx.link'
    link_file = script_dir / link_file_name

    if not link_file.exists():
        print(f"Error: Link file not found: {link_file}")
        print(f"Expected format in link file: <URL> -o <filename>")
        return False

    # Parse link file
    print("📄 Parsing Link File:")
    print("=" * 80)
    print(f"Link file: {link_file}")

    download_url, model_filename = parse_link_file(link_file)

    if download_url is None or model_filename is None:
        return False

    print(f"Download URL: {download_url}")
    print(f"Model name (from .link file): {model_filename}")

    # The final output uses the name from .link file
    final_output = script_dir / model_filename

    # Use temporary name for intermediate download (will be cleaned up)
    temp_download = script_dir / f".tmp_{model_filename}"

    print(f"Final output model: {final_output.name}")

    # Determine which file to use as source for shape fixing
    if args.skip_download:
        # User wants to fix existing model, use it directly if it exists
        if not final_output.exists():
            print(f"\n✗ Error: Model file not found: {final_output}")
            print(f"   Run without --skip-download to download it first.")
            return False
        model_path = final_output
        print(f"Using existing model: {model_path.name}")
    elif final_output.exists() and not args.force_download:
        # Final ONNX already present — skip download
        file_size = final_output.stat().st_size
        print(f"\n⏭  Skipping download :")
        model_path = final_output
        print(f"   {final_output.name} already exists "
              f"({file_size:,} bytes / {file_size / 1024 / 1024:.2f} MB).")
        print(f"   Use --force-download to re-download and re-convert.")
    else:
        # Download to temporary location
        model_path = temp_download

        # Step 1: Download model (unless skipped)
        success = download_model(download_url, model_path, force=args.force_download)
        if not success:
            print("\n✗ Download failed, aborting.")
            # Clean up temporary file if download failed
            if model_path.exists():
                model_path.unlink()
            return False

    # Step 2: Fix shapes (output directly to final location)
    success = fix_model_shape(
        model_path,
        final_output,
        batch_size=args.batch_size,
        channels=args.channels,
        height=args.height,
        width=args.width,
        use_simplifier=not args.no_simplifier
    )

    if success:
        # Step 3: Clean up intermediate file (unless --keep-intermediate)
        if not args.skip_download and model_path != final_output:
            if args.keep_intermediate:
                print(f"\n📦 Keeping intermediate file: {model_path.name}")
            else:
                print(f"\n🗑️  Cleaning up intermediate file...")
                try:
                    model_path.unlink()
                    print(f"✓ Removed: {model_path.name}")
                except Exception as e:
                    print(f"⚠ Could not remove intermediate file: {e}")

        print("\n" + "=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print(f"Final model:    {final_output.name}")
        print(f"Location:       {script_dir}")
        print(f"Input shape:    [{args.batch_size}, {args.channels}, {args.height}, {args.width}]")
        return True
    else:
        print("\n✗ Shape fixing failed")
        # Clean up temporary file on failure
        if not args.skip_download and model_path.exists() and model_path != final_output:
            model_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ONNX model and fix shapes for hardware deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported model variants: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x

Examples:
  # Use default .link file and settings
  %(prog)s

  # Specify a model variant by name (uses <variant>.onnx.link automatically)
  %(prog)s --model yolo26s
  %(prog)s --model yolo26m
  %(prog)s --model yolo26l
  %(prog)s --model yolo26x

  # Specify multiple models to process
  %(prog)s --model yolo26n yolo26s yolo26m
  %(prog)s --model yolo26l yolo26x

  # Prepare every supported model variant
  %(prog)s --model all

  # List all supported model variants and their local status
  %(prog)s --list-models

  # Specify custom .link file explicitly
  %(prog)s --link-file yolo26s.onnx.link

  # Custom shape dimensions
  %(prog)s --batch-size 4 --height 640 --width 640

  # Force re-download
  %(prog)s --force-download

  # Skip model simplification (faster)
  %(prog)s --no-simplifier

  # Skip download (fix existing model only)
  %(prog)s --skip-download

  # Keep intermediate downloaded file
  %(prog)s --keep-intermediate
        """
    )

    parser.add_argument('--model', nargs='+', type=str, default=[DEFAULT_MODEL],
                        choices=YOLO26_VARIANTS + ['all'],
                        metavar='VARIANT',
                        help=f'YOLO26 model variant(s) to prepare. Default: {DEFAULT_MODEL}. '
                             f'One or more of: {", ".join(YOLO26_VARIANTS)}. '
                             f"Use 'all' to prepare every supported variant. "
                             f'Run --list-models to see all options.')
    parser.add_argument('--link-file', type=str, default=None,
                        help='Link file containing download URL '
                             '(default: <model>.onnx.link). Ignored if more than one '
                             '--model variant is specified.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fixed batch size (default: 1)')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of channels (default: 3)')
    parser.add_argument('--height', type=int, default=640,
                        help='Image height (default: 640)')
    parser.add_argument('--width', type=int, default=640,
                        help='Image width (default: 640)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download even if model exists')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, only fix existing model')
    parser.add_argument('--no-simplifier', action='store_true',
                        help='Skip onnx-simplifier optimization')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate downloaded file (not cleaned up)')
    parser.add_argument('--list-models', action='store_true',
                        help='Print all supported model variants and their local status, then exit.')

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent

    if args.list_models:
        list_models(script_dir)
        sys.exit(0)

    if 'all' in args.model:
        args.model = list(YOLO26_VARIANTS)

    # Resolve which variants to process. --link-file overrides the model name
    # only when a single model was requested (matches historical behavior).
    if args.link_file is not None:
        if len(args.model) > 1:
            print("Warning: --link-file is ignored when multiple --model variants are specified")
            models_to_process = args.model
        else:
            models_to_process = [args.link_file.replace('.onnx.link', '')]
    else:
        models_to_process = args.model

    _ensure_dependencies()

    print(f"🚀 Processing {len(models_to_process)} model(s): {', '.join(models_to_process)}")
    print("=" * 80)

    success_count = 0
    for model_variant in models_to_process:
        print(f"\n🔄 Processing model: {model_variant}")
        print("-" * 80)
        if process_single_model(model_variant, script_dir, args):
            success_count += 1
        print(f"\n✅ Completed processing for {model_variant}")
        print("=" * 80)

    print(f"\n🎯 SUMMARY: Successfully processed {success_count}/{len(models_to_process)} model(s)")
    sys.exit(0 if success_count == len(models_to_process) else 1)


if __name__ == "__main__":
    main()
