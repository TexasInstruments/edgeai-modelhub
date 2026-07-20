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
Script to download YOLO11 PyTorch (.pt) model from HuggingFace and convert to ONNX
for hardware deployment on TI edge devices.

Reads .link files containing download URLs and automatically:
 1. Downloads the .pt model if not present
 2. Converts the .pt model to ONNX format using Ultralytics
 3. Fixes dynamic shapes to static shapes
 4. Validates the result

Link file format:
     <download_url> -o <output_filename>

Example:
     https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11n.pt -o yolo11n.pt
"""

import onnx
import sys
import subprocess
from pathlib import Path
from onnx import shape_inference
import argparse


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

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

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


def convert_pt_to_onnx(pt_path, onnx_path, height=640, width=640):
    """
    Convert a YOLO11 PyTorch .pt model to ONNX format using Ultralytics.

    Args:
        pt_path: Path to the input .pt model
        onnx_path: Desired output path for the ONNX model
        height: Input image height
        width: Input image width

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n🔄 Converting .pt to ONNX:")
    print("=" * 80)
    print(f"Input  (.pt):   {pt_path}")
    print(f"Output (.onnx): {onnx_path}")
    print(f"Image size:     {height}x{width}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n✗ ultralytics not installed.")
        print("  Install with: pip install ultralytics")
        return False

    try:
        model = YOLO(str(pt_path))

        # Export to ONNX; Ultralytics places the file next to the .pt by default
        exported = model.export(
            format='onnx',
            imgsz=(height, width),
            dynamic=False,
            simplify=False,   # we run our own simplifier step below
            opset=17,
        )

        exported_path = Path(exported)

        # Move/rename to the desired output path if different
        if exported_path.resolve() != onnx_path.resolve():
            exported_path.rename(onnx_path)
            print(f"\n✓ Moved exported model to: {onnx_path}")

        if onnx_path.exists():
            file_size = onnx_path.stat().st_size
            print(f"✓ ONNX model size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            return True
        else:
            print(f"\n✗ Conversion failed: output file not found at {onnx_path}")
            return False

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        return False


def fix_model_shape(model_path, output_path, batch_size=1, channels=3, height=640, width=640, use_simplifier=True):
    """
    Convert dynamic ONNX model input shape to fixed shape in all layers.

    Uses ONNX shape inference to propagate fixed shapes through all intermediate layers.
    Optionally uses onnxsim for additional simplification and optimization.
    """
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


# Supported YOLO11 model variants
YOLO11_VARIANTS = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']


def process_single_model(model_variant, script_dir, args):
    """
    Process a single model variant.
    
    Args:
        model_variant: Name of the model variant (e.g., 'yolo11n')
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

    download_url, onnx_filename = parse_link_file(link_file)

    if download_url is None or onnx_filename is None:
        return False

    print(f"Download URL: {download_url}")
    print(f"ONNX model name (from .link file): {onnx_filename}")

    # The final output uses the name from .link file
    final_onnx_output = script_dir / onnx_filename

    # Derive the .pt filename from the URL (last path segment, keeping .pt extension)
    pt_filename = Path(download_url.split('?')[0]).name  # e.g. yolo11n.pt
    if not pt_filename.endswith('.pt'):
        pt_filename = Path(onnx_filename).stem + '.pt'
    pt_path = script_dir / pt_filename

    # Temporary ONNX file (before shape fixing)
    temp_onnx_path = script_dir / f".tmp_{onnx_filename}"

    print(f"PT download path:  {pt_path.name}")
    print(f"Final ONNX output: {final_onnx_output.name}")
    print(f"Intermediate ONNX: {temp_onnx_path.name}")

    if args.skip_download:
        # User wants to fix an existing ONNX model
        if not final_onnx_output.exists():
            print(f"\n✗ Error: ONNX model file not found: {final_onnx_output}")
            print(f"   Run without --skip-download to download and convert it first.")
            return False
        print(f"Using existing ONNX model: {final_onnx_output.name}")
        model_path_for_fixing = final_onnx_output
    elif final_onnx_output.exists() and not args.force_download:
        # Final ONNX already present — skip download and conversion entirely
        file_size = final_onnx_output.stat().st_size
        print(f"\n⏭  Skipping download and ONNX conversion:")
        print(f"   {final_onnx_output.name} already exists "
              f"({file_size:,} bytes / {file_size / 1024 / 1024:.2f} MB).")
        print(f"   Use --force-download to re-download and re-convert.")
        model_path_for_fixing = final_onnx_output
    else:
        # Step 1: Download .pt model (always kept)
        success = download_model(download_url, pt_path, force=args.force_download)
        if not success:
            print("\n✗ Download failed, aborting.")
            if pt_path.exists():
                pt_path.unlink()
            return False

        # Step 2: Convert .pt → ONNX to temporary location
        success = convert_pt_to_onnx(
            pt_path,
            temp_onnx_path,
            height=args.height,
            width=args.width,
        )
        if not success:
            print("\n✗ ONNX conversion failed, aborting.")
            # Clean up temporary files
            if temp_onnx_path.exists():
                temp_onnx_path.unlink()
            return False

        print(f"\n📦 PT model kept at: {pt_path.name}")
        model_path_for_fixing = temp_onnx_path

    # Step 3: Fix shapes on the ONNX model (output to final location)
    success = fix_model_shape(
        model_path_for_fixing,
        final_onnx_output,
        batch_size=args.batch_size,
        channels=args.channels,
        height=args.height,
        width=args.width,
        use_simplifier=not args.no_simplifier
    )

    if success:
        # Step 4: Clean up intermediate file (unless --keep-intermediate)
        if not args.skip_download and model_path_for_fixing != final_onnx_output:
            if args.keep_intermediate:
                print(f"\n📦 Keeping intermediate file: {model_path_for_fixing.name}")
            else:
                print(f"\n🗑️  Cleaning up intermediate file...")
                try:
                    model_path_for_fixing.unlink()
                    print(f"✓ Removed: {model_path_for_fixing.name}")
                except Exception as e:
                    print(f"⚠ Could not remove intermediate file: {e}")

        print("\n" + "=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print(f"Final model:    {final_onnx_output.name}")
        print(f"Location:       {script_dir}")
        print(f"Input shape:    [{args.batch_size}, {args.channels}, {args.height}, {args.width}]")
        return True
    else:
        print("\n✗ Shape fixing failed")
        # Clean up temporary file on failure
        if not args.skip_download and model_path_for_fixing.exists() and model_path_for_fixing != final_onnx_output:
            model_path_for_fixing.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download YOLO11 .pt model, convert to ONNX, and fix shapes for hardware deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported model variants: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Notes:
  If the final ONNX file already exists, download and conversion are skipped
  automatically and the existing file is used directly. Pass --force-download
  to override this behaviour and re-download / re-convert.

Examples:
  # Use default .link file and settings (downloads .pt, converts to ONNX, keeps .pt)
  %(prog)s

  # Specify a model variant by name (uses <variant>.onnx.link automatically)
  %(prog)s --model yolo11s
  %(prog)s --model yolo11m
  %(prog)s --model yolo11l
  %(prog)s --model yolo11x

  # Specify multiple models to process
  %(prog)s --models yolo11n yolo11s yolo11m
  %(prog)s --models yolo11l yolo11x

  # Specify custom .link file explicitly
  %(prog)s --link-file yolo11n.onnx.link

  # Custom shape dimensions
  %(prog)s --batch-size 4 --height 640 --width 640

  # Force re-download and re-conversion even if ONNX already exists
  %(prog)s --force-download

  # Skip model simplification (faster)
  %(prog)s --no-simplifier

  # Skip download and conversion (fix existing ONNX model only)
  %(prog)s --skip-download

  # Keep intermediate downloaded file (the temporary ONNX before shape fixing)
  %(prog)s --keep-intermediate
        """
    )

    parser.add_argument('--model', type=str, default=None,
                        choices=YOLO11_VARIANTS,
                        help=f'YOLO11 model variant to prepare. One of: {", ".join(YOLO11_VARIANTS)}. '
                             f'Sets --link-file to <model>.onnx.link automatically. '
                             f'Ignored if --link-file is specified explicitly.')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        choices=YOLO11_VARIANTS,
                        help='List of YOLO11 model variants to prepare. Each model uses <model>.onnx.link. '
                             f'Ignores --model and --link-file.')
    parser.add_argument('--link-file', type=str, default=None,
                        help='Link file containing download URL for the .pt model '
                             '(default: yolo11n.onnx.link, or <model>.onnx.link if --model is set)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fixed batch size (default: 1)')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of channels (default: 3)')
    parser.add_argument('--height', type=int, default=640,
                        help='Image height (default: 640)')
    parser.add_argument('--width', type=int, default=640,
                        help='Image width (default: 640)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download and re-conversion even if the final ONNX already exists')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download and conversion; fix existing ONNX model only')
    parser.add_argument('--no-simplifier', action='store_true',
                        help='Skip onnx-simplifier optimization')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate downloaded file (the temporary ONNX before shape fixing)')

    args = parser.parse_args()

    # Handle mutually exclusive arguments
    if args.models and (args.model or args.link_file):
        print("Warning: --models ignores --model and --link-file arguments")

    # Resolve paths
    script_dir = Path(__file__).parent

    # Process models
    if args.models:
        # Process multiple models
        print(f"🚀 Processing {len(args.models)} models: {', '.join(args.models)}")
        print("=" * 80)
        
        success_count = 0
        for model_variant in args.models:
            print(f"\n🔄 Processing model: {model_variant}")
            print("-" * 80)
            if process_single_model(model_variant, script_dir, args):
                success_count += 1
            print(f"\n✅ Completed processing for {model_variant}")
            print("=" * 80)
        
        print(f"\n🎯 SUMMARY: Successfully processed {success_count}/{len(args.models)} models")
        if success_count == len(args.models):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Process single model (existing behavior)
        # Resolve link file: --link-file takes priority, then --model, then default yolo11n
        if args.link_file is not None:
            link_file_name = args.link_file
        elif args.model is not None:
            link_file_name = f'{args.model}.onnx.link'
        else:
            link_file_name = 'yolo11n.onnx.link'

        # Process the single model using the same function
        model_variant = link_file_name.replace('.onnx.link', '')
        if process_single_model(model_variant, script_dir, args):
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
