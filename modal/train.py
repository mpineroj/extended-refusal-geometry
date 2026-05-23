"""
Run the geometry-of-refusal pipeline (DIM + RDO + Cones + RepInd + Selection) on Modal.

Volume layout (er-geometry-V2):
  /vol/models/<model_name>          — model weights
  /vol/repo/                        — this repo (er-geometry-v2/)
  /vol/results/                     — geometry outputs
  /vol/cache/                       — HuggingFace cache

All artifacts use the same model_name:
  /vol/models/<model_name>
  /vol/results/dim_directions/<model_name>
  /vol/results/rdo/<model_name>

Usage:
  # Full pipeline for baseline (skip DIM since it already exists):
  modal run modal/train.py --model-name Qwen2.5-3B-Instruct --skip-dim

  # Selection only (runs as a separate entrypoint):
  modal run modal/train.py::select --model-name Qwen2.5-3B-Instruct
  modal run modal/train.py::select --model-name Qwen2.5-3B-Instruct-ER-fullweight \\
      --induce-threshold -999
"""

import modal
import os

app = modal.App("geometry-of-refusal")

MODAL_VOLUME_NAME = os.environ.get("GEOMETRY_MODAL_VOLUME", "er-geometry-V2")
vol = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
VOL_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.44.0",
        "accelerate",
        "datasets",
        "nnsight==0.3.6",
        "wandb",
        "python-dotenv",
        "jaxtyping",
        "einops",
        "scipy",
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas",
    )
    .apt_install("git")
)

def _dim_dir(model_name: str) -> str:
    return f"{VOL_PATH}/results/dim_directions/{model_name}"


def _make_env(repo_dir: str) -> dict:
    env = os.environ.copy()
    env.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
        "SAVE_DIR": f"{VOL_PATH}/results",
        "DIM_DIR": "dim_directions",
        "WANDB_MODE": "offline",
        "WANDB_DIR": f"{VOL_PATH}/results/wandb",
        "PYTHONPATH": f"{repo_dir}/src",
    })
    return env


def _write_dotenv(repo_dir: str):
    with open(f"{repo_dir}/.env", "w") as f:
        f.write(f'HUGGINGFACE_CACHE_DIR="{VOL_PATH}/cache"\n')
        f.write(f'SAVE_DIR="{VOL_PATH}/results"\n')
        f.write(f'DIM_DIR="dim_directions"\n')
        f.write(f'WANDB_ENTITY="mpinero-princeton-university"\n')
        f.write(f'WANDB_PROJECT="refusal_directions"\n')
        f.write(f'WANDB_MODE=offline\n')


def _run_cmd(cmd, label, env, cwd):
    import subprocess
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True)
    vol.commit()
    if result.returncode != 0:
        print(f"WARNING: {label} exited with code {result.returncode}")
    return result.returncode


# ── Pipeline ─────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: vol},
    timeout=36000,
)
def run_pipeline(
    model_name: str,
    skip_dim: bool = False,
    skip_rdo: bool = False,
    skip_cones: bool = False,
    skip_repind: bool = False,
):
    """
    Run DIM → RDO → Cones (K=2–5) → RepInd for a single model.

    model_name: model directory name under /vol/models and artifact name under /vol/results.
    """
    import torch

    vol.reload()

    repo_dir = f"{VOL_PATH}/repo"
    model_path = f"{VOL_PATH}/models/{model_name}"
    env = _make_env(repo_dir)
    os.makedirs(f"{VOL_PATH}/results/wandb", exist_ok=True)
    _write_dotenv(repo_dir)

    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    assert os.path.exists(model_path), f"Model not found: {model_path}"

    dim_dir = _dim_dir(model_name)

    # ── 0. DIM ──────────────────────────────────────────────
    if not skip_dim:
        _run_cmd(
            f"python {repo_dir}/src/pipeline/run_pipeline.py "
            f"--model_path {model_path} --no_filter",
            "[0/3] DIM direction extraction",
            env, repo_dir,
        )
        dim_dir = _dim_dir(model_name)

    # Downstream steps need DIM
    if not (skip_rdo and skip_cones and skip_repind):
        assert os.path.exists(f"{dim_dir}/direction.pt"), \
            f"DIM direction not found at {dim_dir}. Run without --skip-dim first."
        print(f"DIM direction: OK ({dim_dir})")

    # ── 1. RDO ──────────────────────────────────────────────
    if not skip_rdo:
        _run_cmd(
            f"python {repo_dir}/src/rdo.py "
            f"--model {model_path} "
            f"--train_direction "
            f"--target_generation_batch_size 32 "
            f"--epochs 1 --lr 0.01 --batch_size 1 --effective_batch_size 16 "
            f"--patience 5 --n_lr_reduce 2 "
            f"--ablation_lambda 1.0 --addition_lambda 0.2 --retain_lambda 1.0",
            "[1/3] RDO direction",
            env, repo_dir,
        )

    # ── 2. Cones (K=2–5) ────────────────────────────────────
    if not skip_cones:
        _run_cmd(
            f"python {repo_dir}/src/rdo.py "
            f"--model {model_path} "
            f"--train_cone "
            f"--target_generation_batch_size 32 "
            f"--min_cone_dim 2 --max_cone_dim 5 "
            f"--epochs 1 --lr 0.01 --batch_size 1 --effective_batch_size 16 "
            f"--patience 5 --n_lr_reduce 2 "
            f"--ablation_lambda 1.0 --addition_lambda 0.2 --retain_lambda 1.0 "
            f"--n_sample 8 --fixed_samples 8",
            "[2/3] Cone training (K=2–5)",
            env, repo_dir,
        )

    # ── 3. RepInd ───────────────────────────────────────────
    if not skip_repind:
        _run_cmd(
            f"python {repo_dir}/src/rdo_repind.py "
            f"--model {model_path} "
            f"--train_independent_direction "
            f"--target_generation_batch_size 32 "
            f"--epochs 2 --lr 0.01 --batch_size 1 --effective_batch_size 16 "
            f"--patience 5 --n_lr_reduce 2 "
            f"--filter_batch_size 4",
            "[3/3] RepInd training",
            env, repo_dir,
        )

    print(f"\n{'=' * 60}\nALL COMPLETE for {model_name}\n{'=' * 60}")


# ── Selection ────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: vol},
    timeout=86400,
)
def run_select(
    model_name: str,
    induce_threshold: float = 0.0,
    skip_existing: bool = False,
    skip_selection: bool = False,
):
    """
    Validation-based selection of RDO/cone candidate pools (select_directions.py).

    Reads vectors from:  results/rdo/<model_name>/vectors/
    Writes selected to:  results/rdo/<model_name>/selected/

    model_name: model directory name under /vol/models and artifact name under /vol/results.
    induce_threshold: 0.0 for baseline (standard refusal); -999 for ER models
    """
    import subprocess

    vol.reload()

    repo_dir = f"{VOL_PATH}/repo"
    model_path = f"{VOL_PATH}/models/{model_name}"
    results_dir = f"{VOL_PATH}/results"

    assert os.path.exists(model_path), f"Model not found: {model_path}"

    if not os.path.exists(f"{results_dir}/rdo/{model_name}/vectors"):
        raise AssertionError(
            f"Vectors dir not found: {results_dir}/rdo/{model_name}/vectors. "
            "Run RDO/cone/RepInd first."
        )
    vectors_dir = f"{results_dir}/rdo/{model_name}/vectors"

    env = os.environ.copy()
    env.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
        "PYTHONPATH": f"{repo_dir}/src",
    })

    skip_arg = "--skip_existing" if skip_existing else ""

    cmd = (
        f"python {repo_dir}/src/select_directions.py "
        f"--model_path {model_path} "
        f"--model_name {model_name} "
        f"--results_dir {results_dir} "
        f"--data_dir {repo_dir}/data "
        f"--induce_threshold {induce_threshold} "
        f"--batch_size 16 "
        f"{skip_arg}"
    )

    print(f"\n{'=' * 60}")
    print(f"Selection: {model_name}")
    print(f"Vectors:   {vectors_dir}")
    print(f"Threshold: induce >= {induce_threshold}")
    print(f"{'=' * 60}\n")

    if skip_selection:
        print("Skipping select_directions.py (--skip-selection); running RepInd copy only.")
    else:
        result = subprocess.run(cmd, shell=True, cwd=repo_dir, env=env, text=True)
        vol.commit()
        if result.returncode != 0:
            print(f"WARNING: selection exited with code {result.returncode}")
            return

    dst = f"{results_dir}/rdo/{model_name}/selected"

    # Copy RepInd vectors (lowest_loss_vector_<run_id>.pt) that have no selected_vector yet.
    # Also write a companion selected_metadata_<run_id>.json so discovery scripts
    # can identify these as RepInd without loading the tensor.
    import json as _json
    repind_vecs_dir = f"{results_dir}/rdo/{model_name}/vectors"
    if os.path.exists(repind_vecs_dir):
        os.makedirs(dst, exist_ok=True)
        for fname in sorted(os.listdir(repind_vecs_dir)):
            if fname.startswith("lowest_loss_vector_") and fname.endswith(".pt"):
                run_id = fname.removeprefix("lowest_loss_vector_").removesuffix(".pt")
                dst_path = os.path.join(dst, f"selected_vector_{run_id}.pt")
                meta_path = os.path.join(dst, f"selected_metadata_{run_id}.json")
                if not os.path.exists(dst_path):
                    shutil.copy2(os.path.join(repind_vecs_dir, fname), dst_path)
                    print(f"RepInd → selected: {run_id}")
                if not os.path.exists(meta_path):
                    with open(meta_path, "w") as f:
                        _json.dump({
                            "run_id": run_id,
                            "model_name": model_name,
                            "k_dim": 1,
                            "shape_1d": True,
                            "method": "repind",
                            "source": fname,
                        }, f, indent=4)

    vol.commit()
    print(f"\nSelection complete for {model_name}")


# ── Entrypoints ──────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model_name: str = "Qwen2.5-3B-Instruct",
    skip_dim: bool = False,
    skip_rdo: bool = False,
    skip_cones: bool = False,
    skip_repind: bool = False,
):
    """
    Run the training pipeline (DIM + RDO + Cones + RepInd).

    Examples:
      # Full pipeline for baseline (DIM exists, skip it):
      modal run modal/train.py --model-name Qwen2.5-3B-Instruct --skip-dim
    """
    run_pipeline.remote(
        model_name=model_name,
        skip_dim=skip_dim,
        skip_rdo=skip_rdo,
        skip_cones=skip_cones,
        skip_repind=skip_repind,
    )


@app.local_entrypoint()
def select(
    model_name: str = "Qwen2.5-3B-Instruct",
    induce_threshold: float = 0.0,
    skip_existing: bool = False,
    skip_selection: bool = False,
):
    """
    Run validation-based direction selection.

    Examples:
      # Baseline (standard refusal, default threshold):
      modal run modal/train.py::select --model-name Qwen2.5-3B-Instruct

      # ER models (disable steering filter):
      modal run modal/train.py::select \\
          --model-name Qwen2.5-3B-Instruct-ER-fullweight \\
          --induce-threshold -999
    """
    run_select.remote(
        model_name=model_name,
        induce_threshold=induce_threshold,
        skip_existing=skip_existing,
        skip_selection=skip_selection,
    )
