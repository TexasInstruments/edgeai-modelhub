#!/usr/bin/env python3
"""
Export DINO classification models (backbone + linear head) from PyTorch Hub
to ONNX with fixed input shapes for TI EdgeAI hardware deployment.

Each exported model includes the full DINO backbone and the pretrained linear
classification head, outputting 1000-class ImageNet logits [1, 1000].

Feature extraction follows DINO's eval_linear.py conventions:
  ViT-S models : last 4 blocks' CLS tokens concatenated → [B, 384×4 = 1536]
  ViT-B models : CLS token + averaged patch tokens (interleaved) → [B, 768×2 = 1536]
  ResNet-50    : avgpool output → [B, 2048]

Supported models (backbone + linear head, 1000-class ImageNet):
  dino_vits16   - ViT-S/16,  21M params, 77.0% linear top-1, 74.5% k-NN top-1
  dino_vits8    - ViT-S/8,   21M params, 79.7% linear top-1, 78.3% k-NN top-1
  dino_vitb16   - ViT-B/16,  85M params, 78.2% linear top-1, 76.1% k-NN top-1
  dino_vitb8    - ViT-B/8,   85M params, 80.1% linear top-1, 77.4% k-NN top-1
  dino_resnet50 - ResNet-50, 23M params, 75.3% linear top-1, 67.5% k-NN top-1

Usage:
  python prepare_model.py --model dino_vits16
  python prepare_model.py --model dino_vitb16 --no-simplifier
  python prepare_model.py --model all
"""

import sys
import subprocess
import tempfile
from pathlib import Path


# n_last_blocks / avgpool follow DINO's eval_linear.py default args per arch:
#   ViT-S: n_last_blocks=4, avgpool=False  → linear_in = 384 * 4 = 1536
#   ViT-B: n_last_blocks=1, avgpool=True   → linear_in = 768 * 2 = 1536
#   ResNet: direct avgpool output          → linear_in = 2048
SUPPORTED_MODELS = {
    'dino_vits16': {
        'arch': 'ViT-S/16', 'params': '21M', 'accuracy_top1': 77.0, 'knn_top1': 74.5,
        'n_last_blocks': 4, 'avgpool': False, 'linear_in': 384 * 4,
    },
    'dino_vits8': {
        'arch': 'ViT-S/8', 'params': '21M', 'accuracy_top1': 79.7, 'knn_top1': 78.3,
        'n_last_blocks': 4, 'avgpool': False, 'linear_in': 384 * 4,
    },
    'dino_vitb16': {
        'arch': 'ViT-B/16', 'params': '85M', 'accuracy_top1': 78.2, 'knn_top1': 76.1,
        'n_last_blocks': 1, 'avgpool': True, 'linear_in': 768 * 2,
    },
    'dino_vitb8': {
        'arch': 'ViT-B/8', 'params': '85M', 'accuracy_top1': 80.1, 'knn_top1': 77.4,
        'n_last_blocks': 1, 'avgpool': True, 'linear_in': 768 * 2,
    },
    'dino_resnet50': {
        'arch': 'ResNet-50', 'params': '23M', 'accuracy_top1': 75.3, 'knn_top1': 67.5,
        'n_last_blocks': None, 'avgpool': False, 'linear_in': 2048,
    },
}

_BASE_URL = 'https://dl.fbaipublicfiles.com/dino/'
LINEAR_WEIGHTS_URLS = {
    'dino_vits16':   _BASE_URL + 'dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth',
    'dino_vits8':    _BASE_URL + 'dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth',
    'dino_vitb16':   _BASE_URL + 'dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth',
    'dino_vitb8':    _BASE_URL + 'dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth',
    'dino_resnet50': _BASE_URL + 'dino_resnet50_pretrain/dino_resnet50_linearweights.pth',
}


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
import torch.nn as nn
import onnx
from onnx import shape_inference
import argparse


class _LinearClassifier(nn.Module):
    """Linear head matching DINO's eval_linear.py LinearClassifier structure."""
    def __init__(self, in_features, num_classes=1000):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class _DinoViTClassifier(nn.Module):
    """
    DINO ViT backbone + linear head for classification.

    Feature extraction matches eval_linear.py:
      - Collects CLS tokens from the last n_last_blocks transformer blocks
      - For ViT-B (avgpool=True): interleaves CLS with averaged patch tokens
        using the same stack+flatten as the original code, preserving weight
        compatibility: [CLS[0], patch[0], CLS[1], patch[1], ...]
    """
    def __init__(self, backbone, linear_head, n_last_blocks, avgpool):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.n = n_last_blocks
        self.avgpool = avgpool

    def forward(self, x):
        intermediate = self.backbone.get_intermediate_layers(x, self.n)
        feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)
        if self.avgpool:
            # Interleave CLS and patch-average as in eval_linear.py:
            # stack → [B, embed, 2] → flatten(1) → [B, embed*2]
            patch_avg = torch.mean(intermediate[-1][:, 1:], dim=1)
            feat = torch.stack([feat, patch_avg], dim=-1).flatten(1)
        return self.linear_head(feat)


class _DinoResNetClassifier(nn.Module):
    """DINO ResNet-50 backbone + linear head for classification."""
    def __init__(self, backbone, linear_head):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head

    def forward(self, x):
        return self.linear_head(self.backbone(x))


def _build_classifier(model_name, info):
    """
    Load DINO backbone from PyTorch Hub, load pretrained linear weights,
    and return a combined classifier module ready for ONNX export.

    Returns the combined nn.Module or None on failure.
    """
    print(f"\nLoading backbone from PyTorch Hub:")
    print(f"  torch.hub.load('facebookresearch/dino:main', '{model_name}')")
    try:
        backbone = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=True)
    except Exception as e:
        print(f"✗ Failed to load backbone: {e}")
        print("  Ensure you have an internet connection and PyTorch installed.")
        return None
    backbone.eval()

    print(f"\nDownloading linear weights:")
    print(f"  URL: {LINEAR_WEIGHTS_URLS[model_name]}")
    try:
        ckpt = torch.hub.load_state_dict_from_url(
            LINEAR_WEIGHTS_URLS[model_name], map_location='cpu', progress=True
        )
        state_dict = ckpt['state_dict']
        # Saved under DDP → strip 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    except Exception as e:
        print(f"✗ Failed to download linear weights: {e}")
        return None

    linear_head = _LinearClassifier(info['linear_in'])
    try:
        linear_head.load_state_dict(state_dict, strict=True)
        print(f"✓ Linear weights loaded ({info['linear_in']} → 1000 classes)")
    except Exception as e:
        print(f"✗ Failed to load linear weights into head: {e}")
        return None
    linear_head.eval()

    if info['n_last_blocks'] is None:
        model = _DinoResNetClassifier(backbone, linear_head)
    else:
        model = _DinoViTClassifier(backbone, linear_head, info['n_last_blocks'], info['avgpool'])

    model.eval()
    return model


def export_to_onnx(model_name, output_path, height=224, width=224):
    """
    Build the DINO backbone + linear head and export to ONNX (opset 17).
    """
    info = SUPPORTED_MODELS[model_name]
    print(f"\nDINO Model Export")
    print("=" * 80)
    print(f"Model:        {model_name} ({info['arch']})")
    print(f"Params:       {info['params']}")
    print(f"Top-1 (lin):  {info['accuracy_top1']}%")
    print(f"Top-1 (k-NN): {info['knn_top1']}%")
    print(f"Input shape:  [1, 3, {height}, {width}]")
    print(f"Output shape: [1, 1000]")

    model = _build_classifier(model_name, info)
    if model is None:
        return False

    dummy_input = torch.randn(1, 3, height, width)

    # Sanity-check output shape before export
    with torch.no_grad():
        out = model(dummy_input)
    if list(out.shape) != [1, 1000]:
        print(f"✗ Unexpected output shape: {list(out.shape)}, expected [1, 1000]")
        return False
    print(f"\n✓ Output shape verified: {list(out.shape)}")

    print(f"\nExporting to ONNX (opset 17):")
    print(f"  Output: {output_path}")

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

            probe = onnx.load(str(tmp_onnx), load_external_data=False)
            has_external = any(
                t.data_location == onnx.TensorProto.EXTERNAL
                for t in probe.graph.initializer
            )

            if has_external:
                # Should not happen for DINO models (<2 GB), but handle gracefully
                print("\nMerging external tensor data...")
                exported = onnx.load(str(tmp_onnx))
            else:
                exported = onnx.load(str(tmp_onnx))
            onnx.save(exported, str(output_path))

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

    if not output_path.exists():
        print("✗ Export failed: output file not created")
        return False

    file_size = output_path.stat().st_size
    print(f"✓ Export completed: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    return True


def fix_model_shape(model_path, output_path, batch_size=1, channels=3, height=224, width=224, use_simplifier=True):
    """
    Convert dynamic ONNX model input shape to fixed shape in all layers.
    """
    print(f"\nFixing Model Shapes:")
    print("=" * 80)
    print(f"Input model:  {model_path}")
    print(f"Output model: {output_path}")

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

    original_shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            original_shape.append(str(dim.dim_value))
        elif dim.dim_param:
            original_shape.append(f"'{dim.dim_param}'")
        else:
            original_shape.append("?")
    print(f"Original shape: [{', '.join(original_shape)}]")

    new_shape = [batch_size, channels, height, width]
    print(f"Fixed shape:    {new_shape}")

    input_tensor.type.tensor_type.shape.ClearField('dim')
    for dim_value in new_shape:
        dim = input_tensor.type.tensor_type.shape.dim.add()
        dim.dim_value = dim_value

    try:
        model = shape_inference.infer_shapes(model)
        print(f"✓ Propagated shapes through {len(model.graph.value_info)} intermediate tensors")
    except Exception as e:
        print(f"⚠ Warning: Shape inference issue: {e}")

    try:
        onnx.checker.check_model(model)
        print("✓ Model validation passed")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False

    if use_simplifier:
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
                orig_nodes = len(graph.node)
                simp_nodes = len(model_simplified.graph.node)
                model = model_simplified
                print(f"✓ Model simplified ({orig_nodes} → {simp_nodes} nodes)")
            else:
                print("⚠ Simplification validation failed, using non-simplified version")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping")
        except Exception as e:
            print(f"⚠ Simplification failed: {e}, continuing without")

    onnx.save(model, str(output_path))
    output_size = output_path.stat().st_size
    print(f"\n✓ Saved: {output_path}  ({output_size / 1024 / 1024:.2f} MB)")

    try:
        verified = onnx.load(str(output_path))
        onnx.checker.check_model(verified)
        for inp in verified.graph.input:
            if any(init.name == inp.name for init in verified.graph.initializer):
                continue
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            if all(isinstance(s, int) and s > 0 for s in shape):
                print(f"✓ Input '{inp.name}': {shape}")
            else:
                print(f"⚠ Input '{inp.name}' has dynamic dimensions: {shape}")
        print("✨ Success! Fixed model ready for deployment")
        return True
    except Exception as e:
        print(f"✗ Final verification failed: {e}")
        return False


def _prepare_single_model(model_name, args, script_dir):
    """Export backbone+head and fix shapes for one model. Returns True on success."""
    info = SUPPORTED_MODELS[model_name]
    final_output = script_dir / f"{model_name}.onnx"

    print(f"\n{'=' * 80}")
    print(f"DINO Model Preparation: {model_name}")
    print(f"{'=' * 80}")
    print(f"Architecture: {info['arch']}  |  Params: {info['params']}")
    print(f"Top-1: {info['accuracy_top1']}%  |  k-NN: {info['knn_top1']}%")
    print(f"Input shape:  [{args.batch_size}, {args.channels}, {args.height}, {args.width}]")

    if args.skip_export:
        if not final_output.exists():
            print(f"\n✗ Error: ONNX file not found: {final_output}")
            print("  Run without --skip-export to export it first.")
            return False
        print(f"\nUsing existing ONNX file: {final_output.name}")
    else:
        if final_output.exists() and not args.force_export:
            print(f"\nONNX file already exists: {final_output.name}")
            print(f"File size: {final_output.stat().st_size / 1024 / 1024:.2f} MB")
            print("Use --force-export to re-export.")
            return True

        success = export_to_onnx(model_name, final_output, args.height, args.width)
        if not success:
            return False

    success = fix_model_shape(
        final_output,
        final_output,
        batch_size=args.batch_size,
        channels=args.channels,
        height=args.height,
        width=args.width,
        use_simplifier=not args.no_simplifier,
    )

    if success:
        print(f"\n{'=' * 80}")
        print("COMPLETE!")
        print(f"{'=' * 80}")
        print(f"Model:       {model_name}")
        print(f"Output:      {final_output.name}  ({final_output.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"Config:      {model_name}_config.yaml")
    else:
        print("\n✗ Shape fixing failed")

    return success


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Export DINO classification models (backbone + linear head) from PyTorch Hub\n'
            'to ONNX format with fixed static shapes for TI EdgeAI hardware deployment.\n'
            '\n'
            'Each model outputs 1000-class ImageNet logits [1, 1000].\n'
            '\n'
            'Processing pipeline:\n'
            '  1. Load pretrained backbone from torch.hub (facebookresearch/dino:main)\n'
            '  2. Download pretrained linear classification weights from Meta AI\n'
            '  3. Combine backbone + linear head into a single module\n'
            '  4. Export to ONNX (opset 17) with dynamic batch axis\n'
            '  5. Fix dynamic input shapes to static [batch, channels, height, width]\n'
            '  6. Run ONNX shape inference and onnxsim simplification\n'
            '  7. Validate the final model'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export default model (ViT-S/16)
  %(prog)s

  # Export a specific variant
  %(prog)s --model dino_vitb16

  # Export all supported models in sequence
  %(prog)s --model all

  # Skip onnxsim (faster, larger output file)
  %(prog)s --model dino_vits16 --no-simplifier

  # Re-run shape inference + onnxsim on an already-exported ONNX file
  %(prog)s --model dino_vits16 --skip-export

  # Force re-export even if ONNX file exists
  %(prog)s --model dino_vits16 --force-export

Available models:
  dino_vits16   - ViT-S/16,  21M params, 77.0%% linear top-1  (recommended for edge)
  dino_vits8    - ViT-S/8,   21M params, 79.7%% linear top-1
  dino_vitb16   - ViT-B/16,  85M params, 78.2%% linear top-1
  dino_vitb8    - ViT-B/8,   85M params, 80.1%% linear top-1
  dino_resnet50 - ResNet-50, 23M params, 75.3%% linear top-1
        """
    )

    parser.add_argument(
        '--model', type=str, default='dino_vits16',
        choices=list(SUPPORTED_MODELS.keys()) + ['all'],
        help='Model variant to export, or "all" to export every model (default: dino_vits16)',
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
        models = list(SUPPORTED_MODELS.keys())
        print(f"Exporting {len(models)} DINO models...")
        results = {}
        for model_name in models:
            results[model_name] = _prepare_single_model(model_name, args, script_dir)

        print(f"\n{'=' * 80}")
        print("ALL MODELS SUMMARY")
        print(f"{'=' * 80}")
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
