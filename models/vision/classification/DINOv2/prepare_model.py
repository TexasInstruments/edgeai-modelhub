#!/usr/bin/env python3
"""
Export DINOv2 classification models (backbone + linear head) from PyTorch Hub
to ONNX with fixed input shapes for TI EdgeAI hardware deployment.

Supported models (backbone + linear classification head, 1000-class ImageNet):
  dinov2_vits14_lc     - ViT-S/14 distilled, 21M params,   81.1% top-1
  dinov2_vitb14_lc     - ViT-B/14 distilled, 86M params,   84.5% top-1
  dinov2_vitl14_lc     - ViT-L/14 distilled, 307M params,  86.3% top-1
  dinov2_vitg14_lc     - ViT-g/14,           1100M params,  86.5% top-1
  dinov2_vits14_reg_lc - ViT-S/14 + registers, 21M params,  80.9% top-1
  dinov2_vitb14_reg_lc - ViT-B/14 + registers, 86M params,  84.6% top-1
  dinov2_vitl14_reg_lc - ViT-L/14 + registers, 307M params, 86.7% top-1
  dinov2_vitg14_reg_lc - ViT-g/14 + registers, 1100M params, 87.1% top-1

Usage:
  python prepare_model.py --model dinov2_vits14_lc
  python prepare_model.py --model dinov2_vitb14_lc --no-simplifier
  python prepare_model.py --model dinov2_vitl14_lc --skip-export
"""

import sys
import subprocess
import tempfile
from pathlib import Path

SUPPORTED_MODELS = {
    'dinov2_vits14_lc':     {'params': '21M',   'accuracy_top1': 81.1, 'gflops': 4.6,   'edge_suitable': True},
    'dinov2_vitb14_lc':     {'params': '86M',   'accuracy_top1': 84.5, 'gflops': 17.6,  'edge_suitable': True},
    'dinov2_vitl14_lc':     {'params': '307M',  'accuracy_top1': 86.3, 'gflops': 61.6,  'edge_suitable': False},
    'dinov2_vitg14_lc':     {'params': '1100M', 'accuracy_top1': 86.5, 'gflops': 314.0, 'edge_suitable': False},
    'dinov2_vits14_reg_lc': {'params': '21M',   'accuracy_top1': 80.9, 'gflops': 4.6,   'edge_suitable': True},
    'dinov2_vitb14_reg_lc': {'params': '86M',   'accuracy_top1': 84.6, 'gflops': 17.6,  'edge_suitable': True},
    'dinov2_vitl14_reg_lc': {'params': '307M',  'accuracy_top1': 86.7, 'gflops': 61.6,  'edge_suitable': False},
    'dinov2_vitg14_reg_lc': {'params': '1100M', 'accuracy_top1': 87.1, 'gflops': 314.0, 'edge_suitable': False},
}

# Models exported with --model all (edge-suitable only; ViT-g excluded due to ~16GB RAM requirement)
EDGE_MODELS = [name for name, info in SUPPORTED_MODELS.items() if info['edge_suitable']]


def _ensure_dependencies():
    required = {
        'onnx': 'onnx',
        'onnxsim': 'onnx-simplifier',
        'torch': 'torch',
    }
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing missing dependency: {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


_ensure_dependencies()

import torch
import onnx
from onnx import shape_inference
import argparse


def _consolidate_external_data(src_onnx_path, dst_onnx_path):
    """
    Consolidate scattered per-tensor external data files into a single .data file
    by streaming — never loads all tensor weights into RAM at once.

    Used for models > 2 GB where onnx.load() would cause an OOM kill.
    Produces two files: dst_onnx_path (proto) and dst_onnx_path.name + '.data'.
    """
    from onnx import TensorProto, AttributeProto

    src_dir = src_onnx_path.parent
    data_filename = dst_onnx_path.name + '.data'
    data_path = dst_onnx_path.parent / data_filename

    # Load proto structure only — tensor data stays on disk
    model = onnx.load(str(src_onnx_path), load_external_data=False)

    def _external_tensors(model_proto):
        for t in model_proto.graph.initializer:
            if t.data_location == TensorProto.EXTERNAL:
                yield t
        for node in model_proto.graph.node:
            for attr in node.attribute:
                if attr.type == AttributeProto.TENSOR and attr.t.data_location == TensorProto.EXTERNAL:
                    yield attr.t
                elif attr.type == AttributeProto.TENSORS:
                    for t in attr.tensors:
                        if t.data_location == TensorProto.EXTERNAL:
                            yield t

    offset = 0
    with open(str(data_path), 'wb') as out_f:
        for tensor in _external_tensors(model):
            info = {e.key: e.value for e in tensor.external_data}
            src_file = src_dir / info['location']
            t_offset = int(info.get('offset', 0))
            t_length = int(info['length'])

            with open(str(src_file), 'rb') as src_f:
                src_f.seek(t_offset)
                out_f.write(src_f.read(t_length))

            del tensor.external_data[:]
            for k, v in [('location', data_filename), ('offset', str(offset)), ('length', str(t_length))]:
                e = tensor.external_data.add()
                e.key = k
                e.value = v
            offset += t_length

    # Save only the proto — tensor data lives in data_path
    onnx.save(model, str(dst_onnx_path))
    print(f"✓ Proto:       {dst_onnx_path.name}")
    print(f"✓ Tensor data: {data_filename}  ({data_path.stat().st_size / 1024**3:.2f} GB)")


def export_to_onnx(model_name, output_path, height=224, width=224):
    """
    Load DINOv2 classifier from PyTorch Hub and export to ONNX.

    torch.onnx.export writes per-tensor external data files alongside its output
    for models exceeding the 2 GB protobuf limit. The export runs inside a
    TemporaryDirectory so those files never appear in the destination directory.

    Small models (S/B/L, < 2 GB): external data is merged inline → single ONNX file.
    Large models (g, > 2 GB):     _consolidate_external_data streams tensors into a
                                   single .data file without loading all weights into RAM.
    """
    info = SUPPORTED_MODELS[model_name]
    print(f"\nLoading model from PyTorch Hub:")
    print("=" * 80)
    print(f"Model:    {model_name}")
    print(f"Params:   {info['params']}")
    print(f"Top-1:    {info['accuracy_top1']}%")
    print(f"GFLOPs:   {info['gflops']}")
    print()

    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("  Ensure you have an internet connection and PyTorch installed.")
        return False

    model.eval()

    # aten::_upsample_bicubic2d_aa (antialias=True) has no opset-17 ONNX handler.
    # Register models set interpolate_antialias=True on the backbone for positional
    # embedding interpolation. Patching it to False uses standard bicubic, which
    # is fully supported and has negligible quality difference at fixed 224x224.
    for module in model.modules():
        if hasattr(module, 'interpolate_antialias'):
            module.interpolate_antialias = False

    dummy_input = torch.randn(1, 3, height, width)

    print(f"\nExporting to ONNX (opset 17):")
    print("-" * 80)
    print(f"Input shape:  [1, 3, {height}, {width}]")
    print(f"Output path:  {output_path}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_onnx = Path(tmpdir) / f"{model_name}.onnx"

            torch.onnx.export(
                model,
                dummy_input,
                str(tmp_onnx),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input':  {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                },
            )

            # Probe proto (no data loaded) to detect external data format
            probe = onnx.load(str(tmp_onnx), load_external_data=False)
            has_external = any(
                t.data_location == onnx.TensorProto.EXTERNAL
                for t in probe.graph.initializer
            )

            if has_external:
                # Model > 2 GB: stream-consolidate without loading all weights into RAM
                print("\nModel exceeds 2 GB — streaming consolidation (no full RAM load)...")
                _consolidate_external_data(tmp_onnx, output_path)
            else:
                # Model fits inline: load and save as single self-contained ONNX
                print("\nMerging external data into single ONNX file...")
                exported = onnx.load(str(tmp_onnx))
                onnx.save(exported, str(output_path))
            # tmpdir and all scattered tensor files are deleted here

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"\n✓ Export completed successfully!")
        print(f"✓ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        return True

    print("✗ Export failed: output file not created")
    return False


def fix_model_shape(model_path, output_path, batch_size=1, channels=3, height=224, width=224, use_simplifier=True):
    """
    Convert dynamic ONNX model input shape to fixed shape in all layers.

    Uses ONNX shape inference to propagate fixed shapes through all intermediate
    layers. Optionally runs onnxsim for additional simplification and optimization.
    """
    print(f"\nFixing Model Shapes:")
    print("=" * 80)
    print(f"Input model:  {model_path}")
    print(f"Output model: {output_path}")

    print(f"\nLoading model...")
    model = onnx.load(str(model_path))

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

    print(f"\nOriginal Shape:")
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

    print(f"\nSetting Fixed Shape:")
    print("-" * 80)
    new_shape = [batch_size, channels, height, width]
    print(f"New shape: {new_shape}")
    print(f"Format: [batch_size, channels, height, width]")

    input_tensor.type.tensor_type.shape.ClearField('dim')
    for dim_value in new_shape:
        dim = input_tensor.type.tensor_type.shape.dim.add()
        dim.dim_value = dim_value

    print(f"\nRunning Shape Inference:")
    print("-" * 80)
    try:
        model = shape_inference.infer_shapes(model)
        value_info_count = len(model.graph.value_info)
        print(f"✓ Propagated shapes through {value_info_count} intermediate tensors")
    except Exception as e:
        print(f"⚠ Warning: Shape inference issue: {e}")
        print("  Continuing with partial inference...")

    print(f"\nValidating Model:")
    print("-" * 80)
    try:
        onnx.checker.check_model(model)
        print("✓ Model validation passed")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False

    if use_simplifier:
        print(f"\nRunning ONNX Simplifier:")
        print("-" * 80)
        try:
            import onnxsim
            model_simplified, check = onnxsim.simplify(
                model,
                check_n=3,
                perform_optimization=True,
                skip_fuse_bn=False,
                overwrite_input_shapes={input_tensor.name: new_shape},
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
        except Exception as e:
            print(f"⚠ Simplification failed: {e}")
            print("  Continuing with non-simplified model")

    print(f"\nSaving Fixed Model:")
    print("-" * 80)
    onnx.save(model, str(output_path))
    output_size = output_path.stat().st_size
    print(f"✓ Saved to: {output_path}")
    print(f"✓ File size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")

    print(f"\nFinal Verification:")
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


def _prepare_single_model(model_name, args, script_dir):
    """Export and fix shapes for one model. Returns True on success."""
    onnx_filename = f"{model_name}.onnx"
    final_output = script_dir / onnx_filename

    info = SUPPORTED_MODELS[model_name]
    if not info['edge_suitable']:
        print(f"\n⚠  Warning: {model_name} has {info['params']} parameters (~"
              f"{int(info['gflops'])} GFLOPs).")
        print(f"   This model requires ~16 GB+ RAM and is not suitable for edge deployment.")
        print(f"   Shape inference and onnxsim will be skipped to reduce memory usage.")
        print()

    print(f"\nDINOv2 Model Preparation")
    print("=" * 80)
    print(f"Model:        {model_name}")
    print(f"Output:       {onnx_filename}")
    print(f"Input shape:  [{args.batch_size}, {args.channels}, {args.height}, {args.width}]")

    if args.skip_export:
        if not final_output.exists():
            print(f"\n✗ Error: ONNX file not found: {final_output}")
            print("  Run without --skip-export to export it first.")
            return False
        print(f"\nUsing existing ONNX file: {final_output.name}")
    else:
        if final_output.exists() and not args.force_export:
            print(f"\nONNX file already exists: {final_output}")
            file_size = final_output.stat().st_size
            print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            print("Use --force-export to re-export.")
            return True

        success = export_to_onnx(model_name, final_output, args.height, args.width)
        if not success:
            return False

    # Disable onnxsim for non-edge models (>2 GB) to avoid OOM during shape fixing
    use_simplifier = not args.no_simplifier and info['edge_suitable']
    success = fix_model_shape(
        final_output,
        final_output,
        batch_size=args.batch_size,
        channels=args.channels,
        height=args.height,
        width=args.width,
        use_simplifier=use_simplifier,
    )

    if success:
        print("\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print(f"Model:        {model_name}")
        print(f"Output:       {final_output.name}")
        print(f"Location:     {script_dir}")
        print(f"Input shape:  [{args.batch_size}, {args.channels}, {args.height}, {args.width}]")
        print(f"Config:       {model_name}_config.yaml")
    else:
        print("\n✗ Shape fixing failed")

    return success


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Export DINOv2 classification models (backbone + linear head) from PyTorch Hub\n'
            'to ONNX format with fixed static shapes for TI EdgeAI hardware deployment.\n'
            '\n'
            'Processing pipeline:\n'
            '  1. Load pretrained model from torch.hub (facebookresearch/dinov2)\n'
            '  2. Export to ONNX (opset 17) with dynamic batch axis\n'
            '  3. Fix dynamic input shapes to static [batch, channels, height, width]\n'
            '  4. Run ONNX shape inference to propagate fixed shapes through all layers\n'
            '  5. Run ONNX simplification via onnxsim (use --no-simplifier to skip)\n'
            '  6. Validate the final model with onnx.checker'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export default model (ViT-S/14)
  %(prog)s

  # Export a specific variant
  %(prog)s --model dinov2_vitb14_lc

  # Export all supported models in sequence
  %(prog)s --model all

  # Export all models, skip onnxsim
  %(prog)s --model all --no-simplifier

  # Export with registers variant
  %(prog)s --model dinov2_vits14_reg_lc

  # Skip onnxsim (faster export, larger model file, shape inference still runs)
  %(prog)s --model dinov2_vitl14_lc --no-simplifier

  # Re-run shape inference + onnxsim on an already-exported ONNX file
  %(prog)s --model dinov2_vits14_lc --skip-export

  # Force re-export even if ONNX file exists
  %(prog)s --model dinov2_vits14_lc --force-export

Available models:
  all                    Export all models listed below in sequence
  dinov2_vits14_lc       ViT-S/14 distilled, 21M params,    81.1%% top-1  (recommended)
  dinov2_vitb14_lc       ViT-B/14 distilled, 86M params,    84.5%% top-1
  dinov2_vitl14_lc       ViT-L/14 distilled, 307M params,   86.3%% top-1
  dinov2_vitg14_lc       ViT-g/14,           1100M params,  86.5%% top-1  (not for edge)
  dinov2_vits14_reg_lc   ViT-S/14 + registers, 21M params,  80.9%% top-1  (recommended)
  dinov2_vitb14_reg_lc   ViT-B/14 + registers, 86M params,  84.6%% top-1
  dinov2_vitl14_reg_lc   ViT-L/14 + registers, 307M params, 86.7%% top-1
  dinov2_vitg14_reg_lc   ViT-g/14 + registers, 1100M params, 87.1%% top-1 (not for edge)
        """
    )

    parser.add_argument(
        '--model', type=str, default='dinov2_vits14_lc',
        choices=list(SUPPORTED_MODELS.keys()) + ['all'],
        help=(
            'Model variant to export, or "all" to export every supported model '
            '(default: dinov2_vits14_lc)'
        ),
    )
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fixed batch size (default: 1)')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of channels (default: 3)')
    parser.add_argument('--height', type=int, default=224,
                        help='Image height (default: 224)')
    parser.add_argument('--width', type=int, default=224,
                        help='Image width (default: 224)')
    parser.add_argument('--force-export', action='store_true',
                        help='Force re-export even if ONNX file already exists')
    parser.add_argument('--skip-export', action='store_true',
                        help='Skip export, only re-run shape inference + onnxsim on existing ONNX')
    parser.add_argument('--no-simplifier', action='store_true',
                        help='Skip onnx-simplifier (onnxsim) step; shape inference still runs')

    args = parser.parse_args()
    script_dir = Path(__file__).parent

    if args.model == 'all':
        models = EDGE_MODELS
        skipped = [m for m in SUPPORTED_MODELS if m not in EDGE_MODELS]
        print(f"Exporting {len(models)} edge-suitable DINOv2 models...")
        if skipped:
            print(f"Skipping (not for edge, ~16 GB RAM required): {', '.join(skipped)}")
        results = {}
        for model_name in models:
            results[model_name] = _prepare_single_model(model_name, args, script_dir)

        print("\n" + "=" * 80)
        print("ALL MODELS SUMMARY")
        print("=" * 80)
        succeeded = [m for m, ok in results.items() if ok]
        failed    = [m for m, ok in results.items() if not ok]
        for m in succeeded:
            print(f"  ✓ {m}")
        for m in failed:
            print(f"  ✗ {m}")
        print(f"\n{len(succeeded)}/{len(models)} models completed successfully.")
        sys.exit(0 if not failed else 1)
    else:
        ok = _prepare_single_model(args.model, args, script_dir)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
