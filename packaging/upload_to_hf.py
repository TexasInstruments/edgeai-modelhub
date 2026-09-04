#!/usr/bin/env python3
"""
Upload edgeai-modelhub model files to HuggingFace Hub.

Only non-binary metadata files are uploaded by default: configs (.yaml),
scripts (.py), documentation (.md), download pointers (.link), and shell
scripts (.sh). Binary model weights (.onnx, .pth) are always excluded.

On re-upload the script diffs local files against the remote repo:
  - New or modified files are uploaded in a single atomic commit.
  - Local files deleted since the last upload are removed from the remote.
  - Unchanged files are skipped; no commit is created when nothing changed.

Deletion is scoped to files whose extension is in INCLUDE_EXTENSIONS, so
remote files this script doesn't manage (e.g. binary .onnx/.pth weights
uploaded separately) are never touched, even though they're absent from
the local file set.

This script can also push the organization's landing-page card ("org card"),
a README.md living in a dedicated "<org>/README" Space repo — see
https://huggingface.co/docs/hub/organizations-cards. Use --org-card for this;
it only ever touches README.md in that Space and never deletes other files
that may already be there (e.g. images added via the web UI). A brand-new
org card Space is always created Public, since HF only renders the card on
the org's profile page when the Space is Public.

By default, --org-card regenerates packaging/ORG_CARD.md from the repo root
README.md via generate_org_card.py immediately before every push (HF's org
card renderer strips <style>/CSS, so README.md's visual design can't survive
there — plain Markdown is what actually gets pushed). This keeps the two
files from drifting apart without hand-editing both. Pass --org-card-source
to push a specific file instead, bypassing generation entirely.

Usage:
    # Dry-run: preview files without uploading
    python upload_to_hf.py --repo-id myorg/edgeai-modelhub --model resnet --dry-run

    # Upload using HF_TOKEN environment variable (recommended)
    HF_TOKEN=hf_xxx python upload_to_hf.py --repo-id myorg/edgeai-modelhub --model resnet

    # Upload with an explicit token
    python upload_to_hf.py --repo-id myorg/edgeai-modelhub --model resnet --token hf_xxx

    # Make the repo public instead of the default private
    python upload_to_hf.py --repo-id myorg/edgeai-modelhub --model resnet --public

    # Exclude a subdirectory (e.g. legacy scripts)
    python upload_to_hf.py --repo-id myorg/edgeai-modelhub --model resnet --exclude-dir temp

    # Upload a list of models in one run (each uses its own registered repo_id)
    HF_TOKEN=hf_xxx python upload_to_hf.py --model resnet convnext dino

    # Upload every model in MODEL_REGISTRY
    HF_TOKEN=hf_xxx python upload_to_hf.py --model all

    # Regenerate packaging/ORG_CARD.md from README.md and push it
    HF_TOKEN=hf_xxx python upload_to_hf.py --org-card

    # Preview the regenerated org card / diff without uploading (no local writes)
    python upload_to_hf.py --org-card --dry-run

    # Push a specific file as the org card, skipping generation entirely
    HF_TOKEN=hf_xxx python upload_to_hf.py --org-card --org-card-source packaging/ORG_CARD.md

    # Upload models and the org card in the same run
    HF_TOKEN=hf_xxx python upload_to_hf.py --model all --org-card

Note: --repo-id only applies when a single --model is given. With multiple
models (or "all"), each model uploads to its own registered repo_id. The org
card uses --org-card-repo-id instead, independent of --repo-id/--model.
"""

import fnmatch
import hashlib
import os
import re
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Callable, Optional


# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── File-selection constants ───────────────────────────────────────────────────

# Only files whose lowercased suffix appears here will be considered.
INCLUDE_EXTENSIONS: frozenset[str] = frozenset({
    '.py',     # Python scripts
    '.yaml',   # Model / session configs
    '.yml',
    '.md',     # Documentation
    '.link',   # Download pointer files
    '.sh',     # Shell scripts
    '.txt',    # Plain text
    '.json',   # JSON configs
})

# Directory *names* (matched against any path component) whose entire subtree
# is excluded. These are never uploaded regardless of file type.
EXCLUDE_DIRS: frozenset[str] = frozenset({
    '__pycache__',
    '.git',
    '.vscode',
    '.claude',
})

# Remote files that are never deleted, even when absent locally.
# .gitattributes is auto-created by HuggingFace Hub to configure LFS tracking.
PROTECTED_REMOTE_FILES: frozenset[str] = frozenset({'.gitattributes'})

# fnmatch patterns matched against filenames only.
EXCLUDE_FILENAME_PATTERNS: tuple[str, ...] = (
    '.tmp_*',   # Temp files left by prepare_model.py
    '*.pyc',    # Python bytecode
    '*.onnx',   # ONNX models
    '*.pth',    # PyTorch models
)

# Minimal YAML front matter HF requires to treat a repo as a valid Space
# (see https://huggingface.co/docs/hub/spaces-config-reference). Injected in
# upload_org_card() only when the source file doesn't already start with its
# own '---' front matter block — e.g. the repo root README.md is a plain
# HTML document with no front matter at all, so pushing it as-is to the
# "<org>/README" Space fails with "Missing configuration in README".
ORG_CARD_FRONT_MATTER = (
    "---\n"
    "title: TIEdgeAI\n"
    "emoji: 🤖\n"
    "colorFrom: red\n"
    "colorTo: gray\n"
    "sdk: static\n"
    "pinned: false\n"
    "---\n\n"
)


# ── Model registry ─────────────────────────────────────────────────────────────
# Map short names to local directory paths (relative to REPO_ROOT).
# Add new models here as they become ready for upload.

MODEL_REGISTRY: dict[str, dict] = {
    'resnet': {
        'dir':         'models/vision/classification/ResNet',
        'repo_id':     'TIEdgeAI/ResNet-Classification',
        'description': 'ResNet-50 ONNX models for TI EdgeAI deployment',
    },
    'convnext': {
        'dir':         'models/vision/classification/ConvNeXt',
        'repo_id':     'TIEdgeAI/ConvNeXt-Classification',
        'description': 'ConvNeXt ONNX models for TI EdgeAI deployment',
    },
    'dino': {
        'dir':         'models/vision/classification/DINO',
        'repo_id':     'TIEdgeAI/DINO-Classification',
        'description': 'DINO ONNX models for TI EdgeAI deployment',
    },
    'dinov2': {
        'dir':         'models/vision/classification/DINOv2',
        'repo_id':     'TIEdgeAI/DINOv2-Classification',
        'description': 'DINOv2 ONNX models for TI EdgeAI deployment',
    },
    'mobilenetv3': {
        'dir':         'models/vision/classification/MobileNetV3',
        'repo_id':     'TIEdgeAI/MobileNetV3-Classification',
        'description': 'MobileNetV3 ONNX models for TI EdgeAI deployment',
    },
    'vit': {
        'dir':         'models/vision/classification/ViT',
        'repo_id':     'TIEdgeAI/ViT-Classification',
        'description': 'ViT ONNX models for TI EdgeAI deployment',
    },
    'yolo11': {
        'dir':         'models/vision/detection/YOLO11',
        'repo_id':     'TIEdgeAI/YOLO11-Detection',
        'description': 'YOLO11 object detection ONNX models for TI EdgeAI',
    },
    'yolo26': {
        'dir':         'models/vision/detection/YOLO26',
        'repo_id':     'TIEdgeAI/YOLO26-Detection',
        'description': 'YOLO26 object detection ONNX models for TI EdgeAI',
    },
    'yolov8': {
        'dir':         'models/vision/detection/YOLOv8',
        'repo_id':     'TIEdgeAI/YOLOv8-Detection',
        'description': 'YOLOv8 object detection ONNX models for TI EdgeAI',
    },
    'yolox': {
        'dir':         'models/vision/detection/YOLOX',
        'repo_id':     'TIEdgeAI/YOLOX-Detection',
        'description': 'YOLOX object detection ONNX models for TI EdgeAI',
    },
    'rtmdet': {
        'dir':         'models/vision/detection/RTMDet',
        'repo_id':     'TIEdgeAI/RTMDet-Detection',
        'description': 'RTMDet object detection ONNX models for TI EdgeAI',
    },
    'deformable_detr': {
        'dir':         'models/vision/detection/Deformable-DETR',
        'repo_id':     'TIEdgeAI/Deformable-DETR-Detection',
        'description': 'Deformable DETR object detection ONNX models for TI EdgeAI',
    },
    'deimv2': {
        'dir':         'models/vision/detection/DEIMv2',
        'repo_id':     'TIEdgeAI/DEIMv2-Detection',
        'description': 'DEIMv2 object detection ONNX models for TI EdgeAI',
    },
    'detr': {
        'dir':         'models/vision/detection/DETR',
        'repo_id':     'TIEdgeAI/DETR-Detection',
        'description': 'DETR object detection ONNX models for TI EdgeAI',
    },
    'rfdetr': {
        'dir':         'models/vision/detection/RF-DETR',
        'repo_id':     'TIEdgeAI/RF-DETR-Detection',
        'description': 'RF-DETR object detection ONNX models for TI EdgeAI',
    },
    'rtdetrv2': {
        'dir':         'models/vision/detection/RT-DETRv2',
        'repo_id':     'TIEdgeAI/RT-DETRv2-Detection',
        'description': 'RT-DETRv2 object detection ONNX models for TI EdgeAI',
    },
    # 'clip': {
    #     'dir':         'models/vision/classification/clip',
    #     'description': 'CLIP vision-language ONNX models for TI EdgeAI',
    # },
}


# ── File collection ────────────────────────────────────────────────────────────

def collect_files(
    model_dir: Path,
    extra_exclude_dirs: tuple[str, ...] = (),
    strip_prefix: bool = True,
    recursive: bool = True,
) -> list[tuple[Path, str]]:
    """
    Walk model_dir and return (local_path, hf_path) pairs for eligible files.

    When strip_prefix=True (default), hf_path is relative to model_dir so
    files land at the HF repo root (e.g. "README.md").
    When strip_prefix=False, hf_path preserves the full path from REPO_ROOT
    (e.g. "models/vision/classification/resnet/README.md").

    Args:
        model_dir: Root of the model directory to scan.
        extra_exclude_dirs: Additional directory names to skip.
        strip_prefix: Strip the model directory prefix from HF paths.
        recursive: When False, only files directly inside model_dir are
                   collected (subdirectories are ignored entirely).

    Returns:
        Sorted list of (local_path, hf_path) pairs.
    """
    all_exclude_dirs = EXCLUDE_DIRS | set(extra_exclude_dirs)
    results: list[tuple[Path, str]] = []

    pattern = model_dir.rglob('*') if recursive else model_dir.glob('*')
    for path in sorted(pattern):
        if not path.is_file():
            continue

        # Path relative to the model dir (e.g. "temp/fix_shape.py")
        rel = path.relative_to(model_dir)

        # Skip if any intermediate directory component is excluded
        if any(part in all_exclude_dirs for part in rel.parts[:-1]):
            continue

        # Skip by filename pattern
        if any(fnmatch.fnmatch(path.name, pat) for pat in EXCLUDE_FILENAME_PATTERNS):
            continue

        # Skip by extension
        if path.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue

        # strip_prefix lands files at repo root (e.g. "README.md")
        # no-strip-prefix preserves the full path from REPO_ROOT
        base = model_dir if strip_prefix else REPO_ROOT
        hf_path = path.relative_to(base).as_posix()
        results.append((path, hf_path))

    return results


# ── Diff helpers ──────────────────────────────────────────────────────────────

def compute_git_sha1(path: Path) -> str:
    """Compute the git blob SHA1 for a file (matches HuggingFace's stored SHA)."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def get_remote_tree(api, repo_id: str, repo_type: str = 'model') -> Optional[dict[str, str]]:
    """
    Return {hf_path: sha1} for all files currently in the remote repo.

    Returns an empty dict when the repo does not exist yet (all local files
    will be treated as new ADDs). Returns None on other errors, which causes
    the caller to skip the diff and upload all local files unconditionally.

    repo_type: 'model' (default) for MODEL_REGISTRY repos, or 'space' for the
    organization card repo.
    """
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore[no-redef]

    result: dict[str, str] = {}
    try:
        for item in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True):
            if hasattr(item, 'blob_id') and item.blob_id:
                result[item.path] = item.blob_id
        return result
    except RepositoryNotFoundError:
        print(f"  Repository not found: {repo_id} (will be created on upload).")
        return result  # empty — all local files are new ADDs
    except Exception as exc:
        print(f"  Warning: could not read remote file tree: {exc}")
        print("  Diff skipped — all eligible local files will be uploaded.")
        return None


def compute_diff(
    local_files: list[tuple[Path, str]],
    remote_tree: dict[str, str],
    scope_prefix: Optional[str],
) -> tuple[list[tuple[Path, str]], list[str], list[tuple[Path, str]]]:
    """
    Diff local files against the remote tree.

    Args:
        local_files:  (local_path, hf_path) pairs from collect_files().
        remote_tree:  {hf_path: sha1} from get_remote_tree().
        scope_prefix: Only consider remote files whose hf_path starts with this
                      prefix as deletion candidates. Pass None to treat the
                      entire remote repo as in scope (use when strip_prefix=True
                      and the repo is dedicated to a single model).

    Returns:
        to_upload:  Local files that are new or whose content has changed.
        to_delete:  Remote hf_paths within scope, with a tracked extension,
                    absent from the local set.
        unchanged:  Local files whose SHA matches the remote exactly.
    """
    local_hf_paths = {hf for _, hf in local_files}

    # Only consider remote files with an extension this script manages as
    # deletion candidates. This keeps unrelated remote content (e.g. binary
    # model weights uploaded through another workflow) from being deleted
    # just because collect_files() never includes them locally.
    in_scope_remote = {
        p: sha for p, sha in remote_tree.items()
        if (scope_prefix is None or p.startswith(scope_prefix))
        and Path(p).suffix.lower() in INCLUDE_EXTENSIONS
    }

    to_upload: list[tuple[Path, str]] = []
    unchanged: list[tuple[Path, str]] = []
    for local, hf in local_files:
        remote_sha = remote_tree.get(hf)
        if remote_sha is None or compute_git_sha1(local) != remote_sha:
            to_upload.append((local, hf))
        else:
            unchanged.append((local, hf))

    to_delete = [
        p for p in in_scope_remote
        if p not in local_hf_paths and p not in PROTECTED_REMOTE_FILES
    ]

    return to_upload, to_delete, unchanged


# ── Authentication ─────────────────────────────────────────────────────────────

def resolve_token(cli_token: Optional[str]) -> str:
    """
    Return a HuggingFace API token.

    Precedence: --token argument > HF_TOKEN environment variable.
    Exits with an informative message when no token is found.
    """
    token = (cli_token or os.environ.get('HF_TOKEN', '')).strip()
    if not token:
        sys.exit(
            "Error: HuggingFace token not found.\n"
            "  Option 1 (recommended): export HF_TOKEN=hf_...\n"
            "  Option 2:               pass --token hf_...\n"
            "  Get a token at:         https://huggingface.co/settings/tokens"
        )
    return token


# ── Repository management ─────────────────────────────────────────────────────

def ensure_repo(
    api,
    repo_id: str,
    private: bool,
    repo_type: str = 'model',
    space_sdk: Optional[str] = None,
) -> None:
    """
    Confirm a HF repository exists, creating it if necessary.

    repo_type: 'model' (default) for MODEL_REGISTRY repos, or 'space' for the
    organization card repo. space_sdk is only used (and required by the HF
    API) when creating a new repo_type='space' repo, e.g. 'static'.

    Handles both old (< 0.20) and new huggingface_hub import paths for
    RepositoryNotFoundError.
    """
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore[no-redef]

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"  Repository exists:  https://huggingface.co/{repo_id}")
    except RepositoryNotFoundError:
        visibility = "private" if private else "public"
        print(f"  Creating {visibility} {repo_type} repository: {repo_id}")
        create_kwargs = dict(repo_id=repo_id, repo_type=repo_type, private=private)
        if repo_type == 'space':
            create_kwargs['space_sdk'] = space_sdk or 'static'
        api.create_repo(**create_kwargs)
        print(f"  Repository created: https://huggingface.co/{repo_id}")


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_files(
    api,
    to_upload: list[tuple[Path, str]],
    to_delete: list[str],
    unchanged: list[tuple[Path, str]],
    repo_id: str,
    commit_message: str,
    dry_run: bool,
    remote_tree: Optional[dict[str, str]] = None,
    repo_type: str = 'model',
) -> None:
    """
    Upload new/changed files and delete removed files in a single atomic commit.

    Skips the commit entirely when nothing has changed. In dry-run mode,
    prints the full local file listing without uploading (no remote comparison).

    Args:
        api:            Authenticated HfApi instance (ignored in dry-run mode).
        to_upload:      Local files to add or overwrite on the remote.
        to_delete:      Remote hf_paths to delete.
        unchanged:      Local files that already match the remote (skipped).
        repo_id:        Target HuggingFace repo ID (e.g. "myorg/edgeai-modelhub").
        commit_message: Commit message for the upload.
        dry_run:        When True, print the file list without uploading.
        remote_tree:    Remote SHA map, used to label ADD vs MODIFY in the summary.
        repo_type:      'model' (default) for MODEL_REGISTRY repos, or 'space'
                         for the organization card repo.
    """
    if dry_run:
        if remote_tree is None:
            # No remote info — just list all local files
            all_local = to_upload + unchanged
            total_kb = sum(p.stat().st_size for p, _ in all_local) / 1024
            print(f"\nLocal files ({len(all_local)} files, {total_kb:.1f} KB total):")
            print(f"  {'Path in HF repo':<60}  {'Size':>9}")
            print(f"  {'-' * 60}  {'-' * 9}")
            for local_path, hf_path in sorted(all_local, key=lambda x: x[1]):
                size_kb = local_path.stat().st_size / 1024
                print(f"  {hf_path:<60}  {size_kb:>7.1f} KB")
            print("\n[dry-run] No files were uploaded.")
            return

        # Remote available — show full diff
        if not to_upload and not to_delete:
            print(f"\n[dry-run] Nothing to do — {len(unchanged)} file(s) already up to date.")
            return

        total_kb = sum(p.stat().st_size for p, _ in to_upload) / 1024
        print(f"\n[dry-run] Changes that would be applied "
              f"({len(to_upload)} to upload, {len(to_delete)} to delete, "
              f"{len(unchanged)} unchanged):")
        print(f"  {'Op':<6}  {'Path in HF repo':<60}  {'Size':>9}")
        print(f"  {'-' * 6}  {'-' * 60}  {'-' * 9}")
        for local_path, hf_path in sorted(to_upload, key=lambda x: x[1]):
            op = 'ADD' if hf_path not in remote_tree else 'MODIFY'
            size_kb = local_path.stat().st_size / 1024
            print(f"  {op:<6}  {hf_path:<60}  {size_kb:>7.1f} KB")
        for hf_path in sorted(to_delete):
            print(f"  {'DELETE':<6}  {hf_path:<60}")
        if unchanged:
            print(f"  ... and {len(unchanged)} unchanged file(s) that would be skipped.")
        print("\n[dry-run] No files were uploaded.")
        return

    if not to_upload and not to_delete:
        print(f"\nNothing to do — {len(unchanged)} file(s) already up to date on the remote.")
        return

    # Diff summary
    total_kb = sum(p.stat().st_size for p, _ in to_upload) / 1024
    print(f"\nChanges ({len(to_upload)} to upload, {len(to_delete)} to delete, "
          f"{len(unchanged)} unchanged):")
    print(f"  {'Op':<6}  {'Path in HF repo':<60}  {'Size':>9}")
    print(f"  {'-' * 6}  {'-' * 60}  {'-' * 9}")
    for local_path, hf_path in sorted(to_upload, key=lambda x: x[1]):
        op = 'ADD' if (remote_tree is None or hf_path not in remote_tree) else 'MODIFY'
        size_kb = local_path.stat().st_size / 1024
        print(f"  {op:<6}  {hf_path:<60}  {size_kb:>7.1f} KB")
    for hf_path in sorted(to_delete):
        print(f"  {'DELETE':<6}  {hf_path:<60}")

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    operations = [
        CommitOperationAdd(path_in_repo=hf, path_or_fileobj=str(local))
        for local, hf in to_upload
    ] + [
        CommitOperationDelete(path_in_repo=hf)
        for hf in to_delete
    ]

    print(f"\nCommitting: '{commit_message}' ...")
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=commit_message,
    )
    print(f"Done. {len(to_upload)} uploaded, {len(to_delete)} deleted ({total_kb:.1f} KB added).")
    print(f"View at: https://huggingface.co/{repo_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Upload edgeai-modelhub model files to HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--repo-id', default=None, metavar='ORG/REPO',
        help='HuggingFace repository ID (defaults to the model\'s registered repo_id)',
    )
    p.add_argument(
        '--model', nargs='+', default=None, metavar='MODEL',
        choices=sorted(MODEL_REGISTRY) + ['all'],
        help=(
            "One or more models to upload (space-separated), or 'all' to "
            "upload every model in MODEL_REGISTRY. Choices: "
            + ', '.join(sorted(MODEL_REGISTRY)) + ', all'
            + ". Not required if --org-card is given."
        ),
    )
    p.add_argument(
        '--org-card', action='store_true',
        help=(
            "Also push this organization's landing-page card: the README.md "
            "of the dedicated '<org>/README' Space repo (see "
            "https://huggingface.co/docs/hub/organizations-cards). Can be "
            "combined with --model, or used on its own."
        ),
    )
    p.add_argument(
        '--org-card-repo-id', default='TIEdgeAI/README', metavar='ORG/README',
        help="Target Space repo for the organization card (default: %(default)s).",
    )
    p.add_argument(
        '--org-card-source', default=None, metavar='PATH',
        help=(
            "Local markdown file to push as the organization card, relative "
            "to the repo root unless an absolute path is given. If omitted "
            "(the default), packaging/ORG_CARD.md is freshly regenerated "
            "from the repo root README.md via generate_org_card.py before "
            "every push, so the two never drift out of sync."
        ),
    )
    p.add_argument(
        '--token', default=None, metavar='HF_TOKEN',
        help='HuggingFace API token (falls back to $HF_TOKEN env var if omitted)',
    )
    p.add_argument(
        '--public', action='store_true',
        help='Make the repository public (default: private)',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Show which files would be uploaded without actually uploading',
    )
    p.add_argument(
        '--commit-message', default=None, metavar='MSG',
        help='Git commit message for the upload (auto-generated if omitted)',
    )
    p.add_argument(
        '--recurse', action='store_true',
        help='Recursively upload files from all subdirectories (default: flat, top-level only)',
    )
    p.add_argument(
        '--exclude-dir', dest='extra_exclude_dirs',
        action='append', default=[], metavar='DIR',
        help='Extra directory name to skip (repeatable, e.g. --exclude-dir temp)',
    )
    p.add_argument(
        '--strip-prefix', action=argparse.BooleanOptionalAction, default=True,
        help=(
            'Strip the full model directory prefix so files land at the HF repo '
            'root (e.g. "README.md" instead of '
            '"models/vision/classification/resnet/README.md"). Default: on. '
            'Use --no-strip-prefix to preserve the full directory structure.'
        ),
    )
    return p


def resolve_model_names(selected: list[str]) -> list[str]:
    """
    Expand the --model argument into a concrete, ordered, de-duplicated list
    of MODEL_REGISTRY keys. A single 'all' anywhere in the list uploads every
    registered model (in sorted order); otherwise the given names are used,
    preserving the order they were passed in and dropping duplicates.
    """
    if 'all' in selected:
        return sorted(MODEL_REGISTRY)

    seen: set[str] = set()
    names: list[str] = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def upload_model(name: str, args: argparse.Namespace) -> bool:
    """
    Run the full collect/diff/upload flow for a single registered model.

    Returns True on success (including dry-runs and no-op "nothing changed"
    runs), False on failure. Exceptions are caught and reported here rather
    than propagated, so one failing model doesn't abort a multi-model batch.
    """
    info = MODEL_REGISTRY[name]
    model_dir = REPO_ROOT / info['dir']
    repo_id = args.repo_id or info['repo_id']

    if not model_dir.is_dir():
        print(f"Error: Model directory not found: {model_dir}")
        return False

    # ── Print run summary ─────────────────────────────────────────────────────
    print(f"Upload configuration — {name}")
    print("=" * 60)
    print(f"  Model:        {name}  ({info['dir']})")
    print(f"  Repository:   {repo_id}")
    print(f"  Visibility:   {'public' if args.public else 'private'}")
    print(f"  Dry-run:      {args.dry_run}")
    print(f"  Strip prefix: {args.strip_prefix}")
    print(f"  Recursive:    {args.recurse}")
    if args.extra_exclude_dirs:
        print(f"  Exclude dirs: {', '.join(args.extra_exclude_dirs)}")
    print()

    # ── Collect eligible files ────────────────────────────────────────────────
    files = collect_files(
        model_dir,
        extra_exclude_dirs=tuple(args.extra_exclude_dirs),
        strip_prefix=args.strip_prefix,
        recursive=args.recurse,
    )

    if not files:
        print(
            "Error: No eligible files found.\n"
            "  Adjust INCLUDE_EXTENSIONS in this script or check the model directory."
        )
        return False

    # Deletion scope: with strip_prefix=True the repo is dedicated to this model
    # so all remote files are in scope. With strip_prefix=False, only files under
    # the model's directory prefix are candidates for deletion.
    scope_prefix = None if args.strip_prefix else (info['dir'] + '/')

    # ── Dry-run: optionally fetch remote for a richer diff ────────────────────
    if args.dry_run:
        token = (args.token or os.environ.get('HF_TOKEN', '')).strip()
        remote_tree = None
        if token:
            try:
                from huggingface_hub import HfApi
                print("Comparing with remote ...")
                remote_tree = get_remote_tree(HfApi(token=token), repo_id)
            except ImportError:
                pass
        else:
            print("  (No HF_TOKEN found — showing local files only. "
                  "Set HF_TOKEN for a remote diff.)\n")

        if remote_tree is not None:
            to_upload, to_delete, unchanged = compute_diff(files, remote_tree, scope_prefix)
        else:
            to_upload, to_delete, unchanged = files, [], []

        upload_files(None, to_upload, to_delete, unchanged, repo_id,
                     commit_message='', dry_run=True, remote_tree=remote_tree)
        return True

    # ── Check that huggingface_hub is installed ───────────────────────────────
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "Error: huggingface_hub is not installed.\n"
            "  Install it with: pip install huggingface-hub"
        )
        return False

    # ── Resolve authentication token ──────────────────────────────────────────
    token = resolve_token(args.token)

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # ── Ensure the target repo exists ─────────────────────────────────────────
    print("Checking repository ...")
    try:
        ensure_repo(api, repo_id, private=not args.public)
    except Exception as exc:
        print(f"Error accessing HuggingFace repository: {exc}")
        return False

    print("Comparing with remote ...")
    remote_tree = get_remote_tree(api, repo_id)

    if remote_tree is not None:
        to_upload, to_delete, unchanged = compute_diff(files, remote_tree, scope_prefix)
    else:
        to_upload, to_delete, unchanged = files, [], []

    # ── Build commit message ──────────────────────────────────────────────────
    commit_message = args.commit_message or f"Add {name} model files"

    # ── Upload ────────────────────────────────────────────────────────────────
    try:
        upload_files(
            api, to_upload, to_delete, unchanged,
            repo_id=repo_id,
            commit_message=commit_message,
            dry_run=False,
            remote_tree=remote_tree,
        )
    except Exception as exc:
        print(f"Error during upload: {exc}")
        return False

    return True


def compose_org_card_content(source_path: Path) -> tuple[bytes, bool]:
    """
    Read source_path and ensure it carries the YAML front matter HF Spaces
    require (title/emoji/colorFrom/colorTo/sdk/pinned) — without it, HF
    rejects the Space with "Missing configuration in README".

    If the file already starts with a '---' front matter block, it's
    assumed to already be a valid Space config and is returned unchanged.
    Otherwise ORG_CARD_FRONT_MATTER is prepended in memory.

    Returns (content, injected) — injected is True if front matter was
    added, purely for logging; the source file on disk is never modified.
    """
    data = source_path.read_bytes()
    if data.lstrip().startswith(b'---'):
        return data, False
    return ORG_CARD_FRONT_MATTER.encode('utf-8') + data, True


# Matches src="..." / href="..." attribute values, for locating local asset
# references (images, etc.) inside an org card's HTML/Markdown content.
_ASSET_REF_RE = re.compile(r'(src|href)=(["\'])([^"\']+)\2', re.IGNORECASE)


def _is_remote_or_special_ref(ref: str) -> bool:
    """True for absolute URLs, data: URIs, mailto: links, and #anchors."""
    return bool(re.match(r'^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|#)', ref))


def localize_org_card_assets(
    content: bytes, source_dir: Path, repo_id: str,
) -> tuple[bytes, list[tuple[Path, str]]]:
    """
    Rewrite local relative asset references (src="..."/href="...") in
    `content` to absolute HF "resolve" URLs for `repo_id`, so images/links
    still work once README.md is pushed to a different repo (the org card
    Space) than the one the source file actually lives in — a plain
    relative path like "docs/assets/logo.png" only resolves correctly in
    the git repo it was written for, not in an arbitrary Space repo that
    doesn't contain that file at all.

    Returns the rewritten content and a list of (local_path, hf_path) pairs
    for every referenced file that actually exists on disk — the caller is
    responsible for uploading these alongside README.md so the rewritten
    URLs actually resolve. References that are already absolute (http(s)://,
    data:, mailto:, #anchors), that don't resolve to a real local file (e.g.
    a link to another folder in this repo that doesn't exist in the Space),
    or that resolve outside REPO_ROOT (e.g. via "../../.." past the repo)
    are left untouched and not collected.

    Each reference is resolved relative to source_dir (the source file's own
    directory — normal relative-link semantics), but the uploaded hf_path is
    always computed relative to REPO_ROOT, so a source file living anywhere
    under the repo (e.g. packaging/ORG_CARD.md referencing
    "../docs/assets/logo.png") lands its asset at the same predictable
    "docs/assets/logo.png" path in the Space regardless of where the source
    file itself sits.
    """
    text = content.decode('utf-8')
    assets: list[tuple[Path, str]] = []

    def _replace(m: re.Match) -> str:
        attr, quote, ref = m.group(1), m.group(2), m.group(3)
        if _is_remote_or_special_ref(ref):
            return m.group(0)

        local_path = (source_dir / ref).resolve()
        if not local_path.is_file():
            return m.group(0)

        try:
            hf_path = local_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            print(f"  Warning: asset '{ref}' resolves outside the repo ({REPO_ROOT}) — leaving link as-is.")
            return m.group(0)

        assets.append((local_path, hf_path))
        resolve_url = f"https://huggingface.co/spaces/{repo_id}/resolve/main/{hf_path}"
        return f'{attr}={quote}{resolve_url}{quote}'

    text = _ASSET_REF_RE.sub(_replace, text)
    return text.encode('utf-8'), assets


def _diff_against_remote(
    files: list[tuple[Path, str]], remote_tree: Optional[dict[str, str]],
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """
    Split staged (local_path, hf_path) pairs into (to_upload, unchanged)
    against remote_tree, with no deletion side — used for the org card,
    which must never delete files it doesn't explicitly manage.
    """
    to_upload: list[tuple[Path, str]] = []
    unchanged: list[tuple[Path, str]] = []
    for local, hf in files:
        remote_sha = remote_tree.get(hf) if remote_tree else None
        if remote_sha == compute_git_sha1(local):
            unchanged.append((local, hf))
        else:
            to_upload.append((local, hf))
    return to_upload, unchanged


def upload_org_card(args: argparse.Namespace) -> bool:
    """
    Push a local markdown file as this organization's landing-page card.

    The org card lives in a dedicated Space repo named "<org>/README"
    (repo_type 'space', sdk 'static') — see
    https://huggingface.co/docs/hub/organizations-cards. Only README.md
    itself is touched: to_delete is intentionally always empty here, so
    other files that may already exist in the Space (e.g. images added via
    the web UI) are never removed just because this script doesn't know
    about them.

    A brand-new org card Space is always created Public — HF only renders
    the card on the org's profile page when the Space is Public, so a
    private one would silently do nothing. Returns True on success
    (including dry-runs and no-op "already up to date" runs), False on
    failure.

    When args.org_card_source is None (the default), packaging/ORG_CARD.md
    is freshly regenerated in-memory from the repo root README.md via
    generate_org_card.py before diffing/uploading, so the two files can
    never silently drift apart. It's only written back to disk on an actual
    (non-dry-run) push — --dry-run never touches local files, it only
    previews. Pass --org-card-source explicitly to push a literal file
    instead, bypassing generation entirely.
    """
    repo_id = args.org_card_repo_id
    auto_generate = args.org_card_source is None

    if auto_generate:
        try:
            import generate_org_card
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import generate_org_card

        source_path = generate_org_card.ORG_CARD_PATH
        source_dir = source_path.parent
        try:
            readme_html = generate_org_card.README_PATH.read_text(encoding='utf-8')
            content = generate_org_card.build_markdown(readme_html).encode('utf-8')
        except (OSError, ValueError) as exc:
            print(f"Error regenerating org card from README.md: {exc}")
            return False
        injected = False
        display_path = f"{source_path.relative_to(REPO_ROOT)}  (regenerated from README.md)"
    else:
        source_path = Path(args.org_card_source)
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()

        if not source_path.is_file():
            print(f"Error: org card source file not found: {source_path}")
            return False

        source_dir = source_path.parent
        content, injected = compose_org_card_content(source_path)
        try:
            display_path = source_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = source_path

    content, assets = localize_org_card_assets(content, source_dir, repo_id)

    print("Upload configuration — org card")
    print("=" * 60)
    print(f"  Source:       {display_path}")
    print(f"  Repository:   {repo_id}  (space)")
    print(f"  Dry-run:      {args.dry_run}")
    if injected:
        print("  Note:         source has no Space front matter — injecting a default "
              "title/sdk:static config block (required by HF, see spaces-config-reference)")
    if assets:
        print(f"  Assets:       {len(assets)} local file(s) referenced by the source — "
              "will be uploaded alongside README.md:")
        for _, hf_path in assets:
            print(f"                  - {hf_path}")
    print()

    # On a real (non-dry-run) auto-generated push, keep the checked-in
    # packaging/ORG_CARD.md in sync with exactly what's about to be pushed
    # (including rewritten asset URLs). Never done during --dry-run, which
    # must not touch local files, and never done when --org-card-source was
    # given explicitly, since that file is the user's own to manage.
    if auto_generate and not args.dry_run:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(content)
        print(f"  Regenerated {source_path.relative_to(REPO_ROOT)} on disk "
              f"({len(content)} bytes) from README.md.\n")

    # Stage README.md (possibly front-matter-injected, with asset references
    # rewritten to absolute resolve URLs) plus every referenced local asset
    # on disk, so the rest of this function can reuse the same (Path,
    # hf_path) plumbing as upload_model() — without ever writing back to the
    # original source file or asset files.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        staged: list[tuple[Path, str]] = []

        readme_staged = tmp_root / 'README.md'
        readme_staged.write_bytes(content)
        staged.append((readme_staged, 'README.md'))

        for local_asset, hf_path in assets:
            asset_staged = tmp_root / hf_path
            asset_staged.parent.mkdir(parents=True, exist_ok=True)
            asset_staged.write_bytes(local_asset.read_bytes())
            staged.append((asset_staged, hf_path))

        # ── Dry-run: optionally fetch remote for a richer diff ────────────────
        if args.dry_run:
            token = (args.token or os.environ.get('HF_TOKEN', '')).strip()
            remote_tree = None
            if token:
                try:
                    from huggingface_hub import HfApi
                    print("Comparing with remote ...")
                    remote_tree = get_remote_tree(HfApi(token=token), repo_id, repo_type='space')
                except ImportError:
                    pass
            else:
                print("  (No HF_TOKEN found — showing local file only. "
                      "Set HF_TOKEN for a remote diff.)\n")

            to_upload, unchanged = _diff_against_remote(staged, remote_tree)
            upload_files(None, to_upload, [], unchanged, repo_id,
                         commit_message='', dry_run=True, remote_tree=remote_tree)
            return True

        # ── Check that huggingface_hub is installed ───────────────────────────
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print(
                "Error: huggingface_hub is not installed.\n"
                "  Install it with: pip install huggingface-hub"
            )
            return False

        # ── Resolve authentication token ───────────────────────────────────────
        token = resolve_token(args.token)

        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # ── Ensure the target Space exists (always Public — see docstring) ────
        print("Checking repository ...")
        try:
            ensure_repo(api, repo_id, private=False, repo_type='space', space_sdk='static')
        except Exception as exc:
            print(f"Error accessing HuggingFace repository: {exc}")
            return False

        print("Comparing with remote ...")
        remote_tree = get_remote_tree(api, repo_id, repo_type='space')
        to_upload, unchanged = _diff_against_remote(staged, remote_tree)

        commit_message = args.commit_message or "Update organization card"

        # ── Upload ──────────────────────────────────────────────────────────────
        try:
            upload_files(
                api, to_upload, [], unchanged,
                repo_id=repo_id,
                commit_message=commit_message,
                dry_run=False,
                remote_tree=remote_tree,
                repo_type='space',
            )
        except Exception as exc:
            print(f"Error during upload: {exc}")
            return False

    return True


def main() -> None:
    args = build_parser().parse_args()

    if not args.model and not args.org_card:
        sys.exit("Error: nothing to do — pass --model (one or more) and/or --org-card.")

    model_names = resolve_model_names(args.model) if args.model else []

    if len(model_names) > 1 and args.repo_id is not None:
        sys.exit(
            "Error: --repo-id can't be combined with multiple --model values "
            "(each registered model has its own target repository).\n"
            "  Pass a single --model with --repo-id, or omit --repo-id and "
            "let each model use its registered repo_id."
        )

    jobs: list[tuple[str, Callable[[], bool]]] = []
    if args.org_card:
        jobs.append(('org-card', lambda: upload_org_card(args)))
    for name in model_names:
        jobs.append((name, lambda name=name: upload_model(name, args)))

    results: dict[str, bool] = {}
    for i, (label, run) in enumerate(jobs):
        if i > 0:
            print("\n" + "─" * 60 + "\n")
        results[label] = run()

    if len(jobs) > 1:
        print("\n" + "=" * 60)
        print(f"Summary ({sum(results.values())}/{len(results)} succeeded)")
        print("=" * 60)
        for label, ok in results.items():
            print(f"  {'OK  ' if ok else 'FAIL'}  {label}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
