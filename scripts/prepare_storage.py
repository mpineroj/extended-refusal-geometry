"""
Create the storage layout expected by the geometry Modal and scratch scripts.

Examples:
  # Adroit/scratch layout
  python scripts/prepare_storage.py

  # Modal volume layout, when running inside Modal
  python scripts/prepare_storage.py --root /vol
"""

import argparse
import os


def parse_args():
    user = os.environ.get("USER", "mp3687")
    default_root = os.environ.get("GEOMETRY_STORAGE_ROOT", f"/scratch/network/{user}")

    parser = argparse.ArgumentParser(description="Prepare geometry experiment storage directories.")
    parser.add_argument(
        "--root",
        default=default_root,
        help="Storage root. Use /vol for the Modal volume layout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    dirs = [
        "models",
        "cache",
        "results",
        "results/wandb",
        "repo",
    ]

    for rel in dirs:
        path = os.path.join(root, rel)
        os.makedirs(path, exist_ok=True)
        print(path)

    print("\nEnvironment values for this layout:")
    print(f'GEOMETRY_MODELS_DIR="{os.path.join(root, "models")}"')
    print(f'HUGGINGFACE_CACHE_DIR="{os.path.join(root, "cache")}"')
    print(f'SAVE_DIR="{os.path.join(root, "results")}"')
    print('DIM_DIR="dim_directions"')


if __name__ == "__main__":
    main()
