"""
Bootstrap a fresh Modal volume for this repository.

This script runs locally. It creates/uses a Modal volume, uploads a clean copy
of this repo to /repo, creates the storage directories expected by modal/*.py,
and can optionally upload local model checkpoints.

Examples:
  python scripts/bootstrap_modal_volume.py

  python scripts/bootstrap_modal_volume.py \
      --models-dir /scratch/network/$USER/models

"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path


DEFAULT_VOLUME_NAME = "er-geometry-V2"
REPO_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "geometry_results_final",
    "models",
    "results",
    "results_from_modal",
    "venv",
}

EXCLUDED_FILES = {
    ".DS_Store",
    ".env",
}

EXCLUDED_SUFFIXES = {
    ".egg-info",
    ".pyc",
    ".pyo",
    ".swp",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap a Modal volume for er-geometry-v2.")
    parser.add_argument(
        "--volume-name",
        default=os.environ.get("GEOMETRY_MODAL_VOLUME", DEFAULT_VOLUME_NAME),
        help="Modal volume name to create/use.",
    )
    parser.add_argument(
        "--repo-path",
        default="/repo",
        help="Destination path for this repo inside the Modal volume.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Optional local model checkpoint directory to upload to /models.",
    )
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=100,
        help="Skip individual repo files larger than this size.",
    )
    return parser.parse_args()


def _ignore_repo_files(max_file_mb):
    max_bytes = max_file_mb * 1024 * 1024

    def ignore(dirpath, names):
        ignored = []
        for name in names:
            path = Path(dirpath) / name
            if name in EXCLUDED_DIRS or name in EXCLUDED_FILES:
                ignored.append(name)
            elif any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
                ignored.append(name)
            elif path.is_file() and path.stat().st_size > max_bytes:
                ignored.append(name)
        return ignored

    return ignore


def _write_keep_file(batch, keep_file, remote_dir):
    batch.put_file(str(keep_file), f"{remote_dir.rstrip('/')}/.keep")


def _stage_repo(staging_root, max_file_mb):
    staged_repo = staging_root / "repo"
    shutil.copytree(REPO_ROOT, staged_repo, ignore=_ignore_repo_files(max_file_mb))
    return staged_repo


def main():
    args = parse_args()

    import modal

    volume = modal.Volume.from_name(args.volume_name, create_if_missing=True)

    with tempfile.TemporaryDirectory(prefix="er-geometry-modal-") as tmp:
        tmp_path = Path(tmp)
        staged_repo = _stage_repo(tmp_path, args.max_file_mb)
        keep_file = tmp_path / ".keep"
        keep_file.write_text("")

        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(staged_repo), args.repo_path)

            # Ensure the expected volume layout exists even before models/results are uploaded.
            for remote_dir in [
                "/cache",
                "/models",
                "/results",
                "/results/wandb",
            ]:
                _write_keep_file(batch, keep_file, remote_dir)

            if args.models_dir:
                batch.put_directory(args.models_dir, "/models")

    print(f"Bootstrapped Modal volume: {args.volume_name}")
    print(f"Repo uploaded to:          {args.repo_path}")
    print("Storage layout:")
    print("  /cache")
    print("  /models")
    print("  /results")
    print("  /results/wandb")


if __name__ == "__main__":
    main()
