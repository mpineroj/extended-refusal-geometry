"""
Evaluate all selected RDO/cone vectors in parallel on Modal.

Usage:
  modal run modal/eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight

This will:
  1. Scan the volume's selected/ directory to discover all vectors for the model
  2. Launch one GPU per (vector, basis) pair in parallel
  3. Save results to results/rdo/<model>/selected/evals/ on the volume
  4. Print a summary table

Run IDs are discovered automatically from the volume — no manual registry needed.
New runs (RDO, cone, RepInd) appear automatically once modal/train.py::select has run.
Already-evaluated pairs are skipped by default (--no-skip-existing to re-run).
"""

import modal
import os

app = modal.App("geometry-eval")

MODAL_VOLUME_NAME = os.environ.get("GEOMETRY_MODAL_VOLUME", "er-geometry-V2")
vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
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
    )
)

image_cpu = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1")
)


# ── Discovery ────────────────────────────────────────────────

@app.function(image=image_cpu, volumes={VOL_PATH: vol}, timeout=120)
def discover_jobs(model_name: str, skip_existing: bool = True) -> list:
    """
    Scan selected/ directories on the volume and return (vector_file, basis_idx) pairs.

    Reads selected_metadata_<run_id>.json for k_dim/shape_1d when available;
    falls back to loading the tensor to infer shape. Skips pairs that already
    have a completed rdo_eval_summary.json unless skip_existing=False.
    """
    import json
    import torch

    vol.reload()

    selected_dir = f"{VOL_PATH}/results/rdo/{model_name}/selected"

    seen_run_ids = set()
    jobs = []

    if not os.path.exists(selected_dir):
        print(f"Selected directory not found: {selected_dir}")
        return []

    for fname in sorted(os.listdir(selected_dir)):
        if not (fname.startswith("selected_vector_") and fname.endswith(".pt")):
            continue

        run_id = fname.removeprefix("selected_vector_").removesuffix(".pt")
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)

        vec_path = os.path.join(selected_dir, fname)
        meta_path = os.path.join(selected_dir, f"selected_metadata_{run_id}.json")

        # Prefer metadata JSON; fall back to tensor shape inspection.
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            k_dim = meta.get("k_dim", 1)
            shape_1d = meta.get("shape_1d", True)
            method = meta.get("method", "unknown")
        else:
            v = torch.load(vec_path, map_location="cpu", weights_only=True)
            if v.dim() == 1:
                k_dim, shape_1d = 1, True
            elif v.shape[0] == 1:
                k_dim, shape_1d = 1, False
            else:
                k_dim, shape_1d = v.shape[0], True
            method = "unknown"

        basis_indices = ([-1] if shape_1d else [0]) if k_dim == 1 else list(range(k_dim))

        for basis_idx in basis_indices:
            suffix = f"_basis{basis_idx}" if basis_idx >= 0 else ""
            eval_dir = (
                f"{VOL_PATH}/results/rdo/{model_name}/selected/evals/"
                f"selected_vector_{run_id}{suffix}"
            )
            if skip_existing and os.path.exists(f"{eval_dir}/rdo_eval_summary.json"):
                print(f"  skip {run_id} basis={basis_idx} (done)")
                continue

            jobs.append((fname, basis_idx))
            print(f"  queue [{method}] {run_id} basis={basis_idx}")

    print(f"\n{len(jobs)} job(s) to launch for {model_name}")
    return jobs


# ── Evaluation ───────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: vol},
    timeout=3600,
)
def eval_one_vector(model_name: str, vector_file: str, basis_idx: int = -1):
    """Evaluate a single selected vector (or one basis vector of a cone) on JailbreakBench."""
    import subprocess
    import json

    vol.reload()

    repo_dir = f"{VOL_PATH}/repo"
    model_path = f"{VOL_PATH}/models/{model_name}"
    vector_path = f"{VOL_PATH}/results/rdo/{model_name}/selected/{vector_file}"
    dim_metadata_path = f"{VOL_PATH}/results/dim_directions/{model_name}/direction_metadata.json"

    vector_stem = vector_file.replace(".pt", "")
    suffix = f"_basis{basis_idx}" if basis_idx >= 0 else ""
    output_dir = f"{VOL_PATH}/results/rdo/{model_name}/selected/evals/{vector_stem}{suffix}"

    with open(f"{repo_dir}/.env", "w") as f:
        f.write(f'HUGGINGFACE_CACHE_DIR="{VOL_PATH}/cache"\n')
        f.write(f'SAVE_DIR="{VOL_PATH}/results"\n')
        f.write(f'DIM_DIR="dim_directions"\n')
        f.write(f'WANDB_MODE=offline\n')

    env = os.environ.copy()
    env.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
        "SAVE_DIR": f"{VOL_PATH}/results",
        "DIM_DIR": "dim_directions",
        "PYTHONPATH": f"{repo_dir}/src",
    })

    cmd = (
        f"python {repo_dir}/src/eval_rdo_direction.py "
        f"--model_path {model_path} "
        f"--vector_path {vector_path} "
        f"--dim_metadata_path {dim_metadata_path} "
        f"--output_dir {output_dir}"
    )
    if basis_idx >= 0:
        cmd += f" --basis_idx {basis_idx}"

    label = f"{vector_file} basis={basis_idx}" if basis_idx >= 0 else vector_file
    print(f"Evaluating: {label}")

    result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-1000:])

    vol.commit()

    summary_path = f"{output_dir}/rdo_eval_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        return {"label": label, "summary": summary}
    else:
        return {"label": label, "error": f"exit code {result.returncode}"}


# ── Entrypoint ───────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model_name: str = "Qwen2.5-3B-Instruct-ER-fullweight",
    skip_existing: bool = True,
):
    """
    Discover and evaluate all selected vectors for a model.

    Examples:
      modal run modal/eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight
      modal run modal/eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight-refusal-only
      modal run modal/eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight-explanation-only
      modal run modal/eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight-justification-only
      modal run modal/eval.py --model-name Qwen2.5-3B-Instruct
      modal run modal/eval.py --model-name Qwen2.5-3B-Instruct --no-skip-existing
    """
    jobs = discover_jobs.remote(model_name, skip_existing)

    if not jobs:
        print("Nothing to evaluate (all done or no vectors found).")
        return

    print(f"\nLaunching {len(jobs)} eval job(s) for {model_name} ...")
    results = list(eval_one_vector.starmap(
        [(model_name, vf, bi) for vf, bi in jobs]
    ))

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — {model_name}")
    print("=" * 70)
    print(f"{'Label':<50} {'Ablation ASR':>12}")
    print("-" * 70)
    for r in results:
        label = r["label"]
        if "summary" in r:
            asr = r["summary"]["results"].get("ablation", "N/A")
            print(f"{label:<50} {asr:>12}")
        else:
            print(f"{label:<50} {'FAILED':>12}")
    print("=" * 70)
