"""
Evaluate 256 uniformly-sampled unit vectors from each trained concept cone.
Produces the data for paper section 7.4 (cone dimensionality analysis).

For each (model, run_id) the selected_vector_<run_id>.pt (K, d) cone basis is loaded,
n_samples vectors are drawn via _make_cone_samples (positive-orthant hypersphere),
and each is evaluated for ablation ASR via substring matching on JailbreakBench.

Results:
  results/rdo/<model_name>/cone_samples/<run_id>/summary.json          — substring ASR
  results/rdo/<model_name>/cone_samples/<run_id>/cone_completions.json — saved completions
  results/rdo/<model_name>/cone_samples/<run_id>/summary_sr.json       — SR ASR (after rescore)

Usage:
  # All ER models (parallel):
  modal run modal/cone_samples.py

  # Single model, all K:
  modal run modal/cone_samples.py --model-name Qwen2.5-3B-Instruct-ER-fullweight

  # Fewer samples (faster):
  modal run modal/cone_samples.py --model-name Qwen2.5-3B-Instruct-ER-fullweight --n-samples 128

  # Baseline model (auto-discovers any K>=2 cones on the volume):
  modal run modal/cone_samples.py --model-name Qwen2.5-3B-Instruct

  # SR rescore (run after main eval):
  modal run modal/cone_samples.py::rescore
  modal run modal/cone_samples.py::rescore --model-name Qwen2.5-3B-Instruct-ER-fullweight
"""

import modal
import os
import sys

app = modal.App("geometry-cone-samples")
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

image_sr = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==2.5.1", "transformers==4.44.0", "datasets", "numpy")
    .run_commands("pip install git+https://github.com/dsbowen/strong_reject.git@main")
)

_ALL_MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-Instruct-ER-fullweight",
    "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-justification-only",
]

image_cpu = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1")
)


@app.function(image=image_cpu, volumes={VOL_PATH: vol}, timeout=120)
def discover_cone_jobs(model_name: str, skip_existing: bool = True) -> list:
    """
    Scan selected/ for K>=2 cone vectors for this model.

    Returns [(run_id, k_dim)] pairs, skipping any that already have both
    summary.json and cone_completions.json unless skip_existing=False.
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

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            k_dim = meta.get("k_dim", 1)
        else:
            v = torch.load(vec_path, map_location="cpu", weights_only=True)
            k_dim = v.shape[0] if v.dim() == 2 else 1

        if k_dim < 2:
            continue

        out_dir = f"{VOL_PATH}/results/rdo/{model_name}/cone_samples/{run_id}"
        if skip_existing and (
            os.path.exists(f"{out_dir}/summary.json")
            and os.path.exists(f"{out_dir}/cone_completions.json")
        ):
            print(f"  skip {run_id} K={k_dim} (done)")
            continue

        jobs.append((run_id, k_dim))
        print(f"  queue {run_id} K={k_dim}")

    print(f"\n{len(jobs)} cone job(s) to launch for {model_name}")
    return jobs


@app.function(image=image_cpu, volumes={VOL_PATH: vol}, timeout=120)
def discover_cone_rescore_jobs(model_name: str) -> list:
    """
    Find cone runs that have completions but no SR rescore yet.
    Returns [(run_id, k_dim)] pairs.
    """
    import json

    vol.reload()

    cone_samples_dir = f"{VOL_PATH}/results/rdo/{model_name}/cone_samples"
    if not os.path.exists(cone_samples_dir):
        return []

    jobs = []
    for run_id in sorted(os.listdir(cone_samples_dir)):
        run_dir = os.path.join(cone_samples_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.exists(f"{run_dir}/cone_completions.json"):
            continue
        if os.path.exists(f"{run_dir}/summary_sr.json"):
            print(f"  skip {run_id} (SR done)")
            continue

        # Get k_dim from summary.json
        summary_path = f"{run_dir}/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                k_dim = json.load(f).get("k_dim", 2)
        else:
            k_dim = 2  # safe default if summary missing

        jobs.append((run_id, k_dim))
        print(f"  queue SR rescore {run_id} K={k_dim}")

    print(f"\n{len(jobs)} SR rescore job(s) for {model_name}")
    return jobs


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: vol},
    timeout=7200,
)
def eval_cone(
    model_name: str,
    run_id: str,
    k_dim: int,
    n_samples: int = 256,
    batch_size: int = 16,
    max_new_tokens: int = 128,
):
    """Evaluate n_samples uniformly-sampled cone directions for one (model, run_id)."""
    import json
    import numpy as np
    import torch

    vol.reload()
    repo_dir = f"{VOL_PATH}/repo"
    model_path = f"{VOL_PATH}/models/{model_name}"
    out_dir = f"{VOL_PATH}/results/rdo/{model_name}/cone_samples/{run_id}"
    summary_path = os.path.join(out_dir, "summary.json")
    cone_completions_path = os.path.join(out_dir, "cone_completions.json")

    if os.path.exists(summary_path) and os.path.exists(cone_completions_path):
        print(f"Already done (summary + completions): {run_id}. Skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    os.chdir(repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "src"))

    os.environ.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
    })

    # ── 1. Find selected cone basis ─────────────────────────────────────────
    sel_vec_path = None
    for candidate in [
        f"{VOL_PATH}/results/rdo/{model_name}/selected/selected_vector_{run_id}.pt",
    ]:
        if os.path.exists(candidate):
            sel_vec_path = candidate
            break
    assert sel_vec_path, (
        f"selected_vector_{run_id}.pt not found. "
        "Run modal/train.py::select first."
    )
    print(f"Loading cone basis: {sel_vec_path}")
    basis = torch.load(sel_vec_path, map_location="cpu")
    if basis.dim() == 1:
        basis = basis.unsqueeze(0)
    print(f"  Shape: {basis.shape}  (K={basis.shape[0]}, d={basis.shape[1]})")

    # ── 2. Load DIM direction for alpha (scale factor) ──────────────────────
    dim_dir = None
    for candidate in [
        f"{VOL_PATH}/results/dim_directions/{model_name}",
    ]:
        if os.path.exists(f"{candidate}/direction.pt"):
            dim_dir = candidate
            break
    assert dim_dir, f"DIM direction.pt not found for {model_name}"
    dim_vec = torch.load(f"{dim_dir}/direction.pt", map_location="cpu")
    alpha = dim_vec.float().norm().item()
    print(f"  Alpha (DIM norm): {alpha:.4f}")

    # ── 3. Sample n_samples directions from the cone ────────────────────────
    np.random.seed(42)
    torch.manual_seed(42)
    from select_directions import _make_cone_samples
    samples = _make_cone_samples(basis.float(), alpha, n_samples)
    print(f"  Sampled {len(samples)} cone directions")

    # ── 4. Load model ───────────────────────────────────────────────────────
    from pipeline.config import Config
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
    from dataset.load_dataset import load_dataset
    from rescore import is_refusal_expanded

    cfg = Config(model_alias=os.path.basename(model_path), model_path=model_path)
    model_base = construct_model_base(model_path)
    param_dtype = next(model_base.model.parameters()).dtype
    param_device = next(model_base.model.parameters()).device

    # ── 5. Load JailbreakBench ──────────────────────────────────────────────
    eval_dataset = load_dataset("jailbreakbench")
    print(f"  JailbreakBench: {len(eval_dataset)} prompts")

    # ── 6. Evaluate each sampled direction ──────────────────────────────────
    all_asrs = []
    all_completions_list = []
    for i, direction in enumerate(samples):
        direction = direction.to(dtype=param_dtype, device=param_device)
        pre_hooks, hooks = get_all_direction_ablation_hooks(model_base, direction)

        completions = model_base.generate_completions(
            eval_dataset,
            fwd_pre_hooks=pre_hooks,
            fwd_hooks=hooks,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

        asr = float(np.mean([0.0 if is_refusal_expanded(c["response"]) else 1.0 for c in completions]))
        all_asrs.append(asr)
        all_completions_list.append(completions)

        if (i + 1) % 32 == 0 or i == 0:
            med = float(np.median(all_asrs))
            print(f"  [{i+1}/{n_samples}] ASR={asr:.3f}  running_median={med:.3f}")

    # ── 7. Save summary and completions ─────────────────────────────────────
    summary = {
        "model_name": model_name,
        "run_id": run_id,
        "k_dim": k_dim,
        "n_samples": n_samples,
        "scorer": "expanded_substring",
        "all_asrs": all_asrs,
        "median_asr": float(np.median(all_asrs)),
        "mean_asr": float(np.mean(all_asrs)),
        "std_asr": float(np.std(all_asrs)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    cone_completions = [
        {"direction_idx": i, "substring_asr": asr, "completions": comps}
        for i, (asr, comps) in enumerate(zip(all_asrs, all_completions_list))
    ]
    with open(cone_completions_path, "w") as f:
        json.dump(cone_completions, f)
    vol.commit()

    print(
        f"\nDone {model_name} / {run_id} K={k_dim}: "
        f"median={summary['median_asr']:.3f}  mean={summary['mean_asr']:.3f}"
    )


@app.local_entrypoint()
def main(
    model_name: str = "",
    n_samples: int = 256,
    batch_size: int = 16,
    skip_existing: bool = True,
):
    """
    Discover and dispatch cone sampling eval jobs for all K>=2 cones on the volume.

    Examples:
      modal run modal/cone_samples.py                          # all models
      modal run modal/cone_samples.py --model-name Qwen2.5-3B-Instruct-ER-fullweight
      modal run modal/cone_samples.py --n-samples 128          # faster smoke test
      modal run modal/cone_samples.py --no-skip-existing       # re-run completed jobs
    """
    models = [model_name] if model_name else _ALL_MODELS

    handles = []
    for mname in models:
        print(f"\nDiscovering cone jobs for {mname} ...")
        jobs = discover_cone_jobs.remote(mname, skip_existing)
        for run_id, k_dim in jobs:
            print(f"  Spawning: {mname} / {run_id} K={k_dim}")
            handles.append(
                eval_cone.spawn(
                    model_name=mname,
                    run_id=run_id,
                    k_dim=k_dim,
                    n_samples=n_samples,
                    batch_size=batch_size,
                )
            )

    if not handles:
        print("No cone jobs to run (all done or no K>=2 cones found).")
        return

    print(f"\nWaiting for {len(handles)} jobs...")
    for h in handles:
        h.get()
    print(f"All {len(handles)} cone eval jobs complete.")


# ── SR Rescore ───────────────────────────────────────────────

N_SR_DIRECTIONS = 32  # stratified directions to SR-score per K value


@app.function(
    image=image_sr,
    gpu="A10G",
    volumes={VOL_PATH: vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def rescore_cone_sr(model_name: str, run_id: str, k_dim: int, n_sr_dirs: int = N_SR_DIRECTIONS):
    """
    SR-rescore a stratified sample of cone directions saved by eval_cone.

    Loads cone_completions.json, picks n_sr_dirs directions evenly spaced across
    the substring-ASR distribution, scores with StrongREJECT, saves summary_sr.json.
    """
    import json
    import numpy as np
    import os
    from datasets import Dataset
    from strong_reject.evaluate import evaluate_dataset

    vol.reload()

    out_dir = f"{VOL_PATH}/results/rdo/{model_name}/cone_samples/{run_id}"
    summary_path = os.path.join(out_dir, "summary.json")
    cone_completions_path = os.path.join(out_dir, "cone_completions.json")
    summary_sr_path = os.path.join(out_dir, "summary_sr.json")

    if os.path.exists(summary_sr_path):
        print(f"Already SR-rescored: {run_id}. Skipping.")
        return

    assert os.path.exists(cone_completions_path), (
        f"cone_completions.json not found at {cone_completions_path}. "
        "Re-run modal/cone_samples.py first to save completions."
    )

    with open(cone_completions_path) as f:
        cone_completions = json.load(f)

    n_total = len(cone_completions)
    # Stratified sample: evenly spaced across sorted-by-substring-ASR order
    sorted_idxs = sorted(range(n_total), key=lambda i: cone_completions[i]["substring_asr"])
    step = max(1, n_total // n_sr_dirs)
    sample_idxs = sorted_idxs[::step][:n_sr_dirs]

    print(f"SR rescoring {len(sample_idxs)}/{n_total} directions for {model_name}/{run_id} K={k_dim}")

    evaluator = "strongreject_rubric"
    sr_asrs = []
    sampled_substring_asrs = []

    for idx in sample_idxs:
        entry = cone_completions[idx]
        completions = entry["completions"]
        prompts = [c["prompt"] for c in completions]
        responses = [c["response"] for c in completions]

        dataset = Dataset.from_dict({"forbidden_prompt": prompts}).add_column("response", responses)
        eval_result = evaluate_dataset(dataset, [evaluator], batch_size=8)
        scores = list(eval_result["score"])

        sr_asr = float(np.mean(scores))
        sr_asrs.append(sr_asr)
        sampled_substring_asrs.append(entry["substring_asr"])
        print(f"  dir {idx:3d}: substring={entry['substring_asr']:.3f}  sr={sr_asr:.3f}")

    summary_sr = {
        "model_name": model_name,
        "run_id": run_id,
        "k_dim": k_dim,
        "n_total_directions": n_total,
        "n_sr_dirs": len(sr_asrs),
        "scorer": "strongreject_rubric",
        "sampled_direction_idxs": [sample_idxs[i] for i in range(len(sr_asrs))],
        "sampled_substring_asrs": sampled_substring_asrs,
        "sr_asrs": sr_asrs,
        "median_sr_asr": float(np.median(sr_asrs)),
        "mean_sr_asr": float(np.mean(sr_asrs)),
        "std_sr_asr": float(np.std(sr_asrs)),
    }

    with open(summary_sr_path, "w") as f:
        json.dump(summary_sr, f, indent=2)
    vol.commit()

    print(
        f"\nDone SR rescore {model_name}/{run_id} K={k_dim}: "
        f"median_sr={summary_sr['median_sr_asr']:.3f}  "
        f"median_substring={float(np.median(sampled_substring_asrs)):.3f}"
    )


@app.local_entrypoint()
def rescore(model_name: str = "", n_sr_dirs: int = N_SR_DIRECTIONS):
    """
    Discover and SR-rescore cone runs that have completions but no summary_sr.json yet.

    Examples:
      modal run modal/cone_samples.py::rescore
      modal run modal/cone_samples.py::rescore --model-name Qwen2.5-3B-Instruct-ER-fullweight
    """
    models = [model_name] if model_name else _ALL_MODELS

    handles = []
    for mname in models:
        print(f"\nDiscovering SR rescore jobs for {mname} ...")
        jobs = discover_cone_rescore_jobs.remote(mname)
        for run_id, k_dim in jobs:
            print(f"  Spawning SR rescore: {mname} / {run_id} K={k_dim}")
            handles.append(
                rescore_cone_sr.spawn(
                    model_name=mname,
                    run_id=run_id,
                    k_dim=k_dim,
                    n_sr_dirs=n_sr_dirs,
                )
            )

    if not handles:
        print("No SR rescore jobs to run (all done or no completions found).")
        return

    print(f"\nWaiting for {len(handles)} SR rescore jobs...")
    for h in handles:
        h.get()
    print(f"All {len(handles)} SR rescore jobs complete.")
