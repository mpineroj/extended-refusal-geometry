"""
Populate the Modal volume with Hugging Face-compatible model directories.

Usage:
  # After scripts/bootstrap_modal_volume.py has uploaded the repo to /vol/repo:
  modal run modal/download_models.py

  # Skip the baseline if it is already present:
  modal run modal/download_models.py --skip-baseline
"""

import os
import subprocess

import modal


app = modal.App("geometry-download-models")

MODAL_VOLUME_NAME = os.environ.get("GEOMETRY_MODAL_VOLUME", "er-geometry-V2")
vol = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
VOL_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.44.0",
        "accelerate",
        "python-dotenv",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: vol},
    timeout=36000,
)
def download(skip_baseline: bool = False):
    vol.reload()

    repo_dir = f"{VOL_PATH}/repo"
    script = f"{repo_dir}/scripts/download_models.py"
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"{script} not found. Run scripts/bootstrap_modal_volume.py first."
        )

    os.makedirs(f"{VOL_PATH}/models", exist_ok=True)
    os.makedirs(f"{VOL_PATH}/cache", exist_ok=True)

    env = os.environ.copy()
    env.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
    })

    cmd = [
        "python",
        script,
        "--models-dir",
        f"{VOL_PATH}/models",
        "--cache-dir",
        f"{VOL_PATH}/cache",
    ]
    if skip_baseline:
        cmd.append("--skip-baseline")

    result = subprocess.run(cmd, cwd=repo_dir, env=env, text=True)
    vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"download_models.py exited with code {result.returncode}")


@app.local_entrypoint()
def main(skip_baseline: bool = False):
    download.remote(skip_baseline=skip_baseline)
