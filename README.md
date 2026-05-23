# ER Geometry V2 Runbook

This repository contains the geometry-analysis code used to study refusal
directions, refusal cones, and representational independence. It assumes model
checkpoints already exist, either as Hugging Face model IDs or local
Hugging Face-compatible checkpoint directories.

This repo does not train ER models from scratch and does not implement the full
behavioral study pipeline for CatQA, MMLU, C4 perplexity, or TwinBreak. It
focuses on DIM, RDO, cone, RepInd, selection, evaluation, rescoring, and figure
generation.

## Supported Model Paths

The geometry code supports model families with wrappers/templates in `src/`:

- Qwen/Qwen2.5-style models
- Gemma instruction models
- Llama-3 instruction models

Model paths may be either:

- Hugging Face IDs, for local runs, for example `Qwen/Qwen2.5-3B-Instruct`
- Local checkpoint directories, for local or Modal runs, for example
  `/vol/models/Qwen2.5-3B-Instruct-ER-fullweight`

For Modal, the existing runners expect model directories at:

```text
/vol/models/<model_name>
```

where `<model_name>` is the argument passed to `--model-name`.

## Fresh Local Environment

Use Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some optional scripts require extra packages that are not part of the core
geometry install:

```bash
pip install peft
pip install git+https://github.com/dsbowen/strong_reject.git@main
```

Create `.env`:

```bash
cp .env.example .env
```

Set at least:

```bash
HUGGINGFACE_CACHE_DIR="/path/to/hf_cache"
SAVE_DIR="/path/to/results"
DIM_DIR="dim_directions"
WANDB_PROJECT="refusal_directions"
WANDB_MODE=offline
```

Prepare local storage directories:

```bash
python scripts/prepare_storage.py --root /path/to/storage
```

## Local Geometry Workflow

Set `PYTHONPATH` when running scripts from the repo root:

```bash
export PYTHONPATH="$PWD/src"
```

### 1. Compute DIM Directions

For a Hugging Face model ID:

```bash
python src/pipeline/run_pipeline.py \
  --model_path Qwen/Qwen2.5-3B-Instruct \
  --no_filter
```

For a local checkpoint:

```bash
python src/pipeline/run_pipeline.py \
  --model_path /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --no_filter
```

Artifacts are written under:

```text
$SAVE_DIR/$DIM_DIR/<model_basename>
```

The downstream RDO scripts need:

```text
direction.pt
direction_metadata.json
generate_directions/mean_diffs.pt
```

### 2. Train RDO Direction

```bash
python src/rdo.py \
  --model /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --train_direction \
  --target_generation_batch_size 32 \
  --epochs 1 \
  --lr 0.01 \
  --batch_size 1 \
  --effective_batch_size 16 \
  --ablation_lambda 1.0 \
  --addition_lambda 0.2 \
  --retain_lambda 1.0
```

Outputs are written under:

```text
$SAVE_DIR/rdo/<model_basename>/vectors/
```

### 3. Train Refusal Cones

```bash
python src/rdo.py \
  --model /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --train_cone \
  --target_generation_batch_size 32 \
  --min_cone_dim 2 \
  --max_cone_dim 5 \
  --epochs 1 \
  --lr 0.01 \
  --batch_size 1 \
  --effective_batch_size 16 \
  --ablation_lambda 1.0 \
  --addition_lambda 0.2 \
  --retain_lambda 1.0 \
  --n_sample 8 \
  --fixed_samples 8
```

### 4. Train RepInd Directions

```bash
python src/rdo_repind.py \
  --model /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --train_independent_direction \
  --target_generation_batch_size 32 \
  --epochs 2 \
  --lr 0.01 \
  --batch_size 1 \
  --effective_batch_size 16 \
  --filter_batch_size 4
```

RepInd currently expects DIM artifacts to live under
`$SAVE_DIR/$DIM_DIR/<model_basename>`.

### 5. Select Candidate Directions

Selection scores candidate pools in `rdo/<model_name>/vectors` and writes
selected vectors to `rdo/<model_name>/selected`.

```bash
python src/select_directions.py \
  --model_path /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --model_name Qwen2.5-3B-Instruct-ER-fullweight \
  --results_dir /path/to/results \
  --data_dir data \
  --batch_size 16
```

For ER variants where activation-addition refusal induction is not expected to
pass the baseline filter, relax the induction threshold:

```bash
python src/select_directions.py \
  --model_path /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --model_name Qwen2.5-3B-Instruct-ER-fullweight \
  --results_dir /path/to/results \
  --data_dir data \
  --induce_threshold -999 \
  --batch_size 16
```

### 6. Evaluate Selected Directions

Evaluate a single selected vector:

```bash
python src/eval_rdo_direction.py \
  --model_path /path/to/models/Qwen2.5-3B-Instruct-ER-fullweight \
  --vector_path /path/to/results/rdo/Qwen2.5-3B-Instruct-ER-fullweight/selected/selected_vector_<run_id>.pt \
  --dim_metadata_path /path/to/results/dim_directions/Qwen2.5-3B-Instruct-ER-fullweight/direction_metadata.json \
  --output_dir /path/to/results/rdo/Qwen2.5-3B-Instruct-ER-fullweight/selected/evals/selected_vector_<run_id>
```

For cone basis vectors, add:

```bash
--basis_idx 0
```

## Modal Workflow

The Modal workflow uses a persistent volume named `er-geometry-V2` by default.
Override it with `GEOMETRY_MODAL_VOLUME` if needed.

### 1. Bootstrap Volume

From your local checkout:

```bash
python scripts/bootstrap_modal_volume.py
```

This uploads a clean copy of the repo to:

```text
/vol/repo
```

and creates:

```text
/vol/cache
/vol/models
/vol/results
/vol/results/wandb
```

If you have local model checkpoints:

```bash
python scripts/bootstrap_modal_volume.py \
  --models-dir /path/to/local/models
```

### 2. Populate Models

The runners require one Hugging Face-compatible directory per model under:

```text
/vol/models/<model_name>
```

The default model names used by the Qwen2.5 ER study are:

```text
Qwen2.5-3B-Instruct
Qwen2.5-3B-Instruct-ER-fullweight
Qwen2.5-3B-Instruct-ER-fullweight-refusal-only
Qwen2.5-3B-Instruct-ER-fullweight-explanation-only
Qwen2.5-3B-Instruct-ER-fullweight-justification-only
```

If the checkpoints are available from Hugging Face, populate a fresh Modal
volume directly:

```bash
modal run modal/download_models.py
```

This writes to:

```text
/vol/models
/vol/cache
```

`scripts/download_models.py` expects each configured Hugging Face repo/subfolder
to be loadable as an `AutoModelForCausalLM`. If a checkpoint is published only
as a PEFT/LoRA adapter, first materialize it as a full checkpoint with
`scripts/merge_lora.py`, then upload the resulting model directory.

If you are running inside another environment with `/vol` mounted, the same
downloader can be called directly:

```bash
python /vol/repo/scripts/download_models.py \
  --models-dir /vol/models \
  --cache-dir /vol/cache
```

If you already have local checkpoints, upload them during bootstrap instead:

```bash
python scripts/bootstrap_modal_volume.py \
  --models-dir /path/to/local/models
```

For other model families or custom checkpoints, use the same layout:

```text
/vol/models/<model_name>
```

and pass that exact directory name to `--model-name`.

### 3. Run Training on Modal

Baseline example:

```bash
modal run modal/train.py \
  --model-name Qwen2.5-3B-Instruct
```

ER example using a full model directory already in `/vol/models`:

```bash
modal run modal/train.py \
  --model-name Qwen2.5-3B-Instruct-ER-fullweight
```

Run only parts of the pipeline:

```bash
modal run modal/train.py \
  --model-name Qwen2.5-3B-Instruct-ER-fullweight \
  --skip-dim \
  --skip-rdo \
  --skip-cones
```

### 4. Select and Evaluate on Modal

Selection:

```bash
modal run modal/train.py::select \
  --model-name Qwen2.5-3B-Instruct-ER-fullweight \
  --induce-threshold -999
```

Evaluate selected vectors:

```bash
modal run modal/eval.py \
  --model-name Qwen2.5-3B-Instruct-ER-fullweight
```

Evaluate cone samples:

```bash
modal run modal/cone_samples.py \
  --model-name Qwen2.5-3B-Instruct-ER-fullweight \
  --n-samples 256
```

Optional StrongREJECT rescore:

```bash
modal run modal/rescore.py \
  --model-name Qwen2.5-3B-Instruct-ER-fullweight \
  --use-strongreject
```

## Generate Final Figures

The main figure/table entrypoint expects `--results-root` to point at a
directory that contains a `results/` subdirectory. For example:

```text
geometry_results_final/results/dim_directions/...
geometry_results_final/results/rdo/...
```

Run:

```bash
python analysis/analyze.py \
  --results-root /path/to/geometry_results_final \
  --output-dir figures
```

Run selected sections only:

```bash
python analysis/analyze.py \
  --results-root /path/to/geometry_results_final \
  --output-dir figures \
  --sections 71,72,73,74,75,78
```

Expected outputs:

```text
figures/7.1_dim_heatmap_grid.png
figures/7.1_dim_heatmap_grid.pdf
figures/7.2_dim_quality_table.csv
figures/7.2_dim_quality_table.txt
figures/7.3_dim_rdo_alignment.png
figures/7.3_dim_rdo_alignment.pdf
figures/7.4_cone_dimensionality.png
figures/7.4_cone_dimensionality.pdf
figures/7.5_causal_load.png
figures/7.5_causal_load.pdf
figures/7.8_dim_vs_rdo.png
figures/7.8_dim_vs_rdo.pdf
figures/summary.json
```

If results live on Modal, download them first with the Modal CLI into a local
directory shaped like `geometry_results_final/results/...`, then point
`--results-root` at `geometry_results_final`:

```bash
modal volume get er-geometry-V2 /results ./geometry_results_final/results
```

## Fresh-Run Checklist

1. Create Python 3.10 environment and install dependencies.
2. Authenticate Modal locally with `modal setup`.
3. Bootstrap `er-geometry-V2` with `scripts/bootstrap_modal_volume.py`.
4. Populate `/vol/models` with `modal/download_models.py` or `--models-dir`.
5. Run DIM extraction for each model.
6. Run RDO, cone, and RepInd training.
7. Select candidate vectors.
8. Evaluate selected vectors and cone samples.
9. Rescore with StrongREJECT if needed.
10. Download `/results` and run `analysis/analyze.py` to produce figures.

## Known Constraints

- `modal/eval.py` and `modal/cone_samples.py` discover selected vectors from
  the volume. The figure script `analysis/analyze.py` still has a fixed Qwen2.5
  ER study registry and known run IDs; add new model names and selected run IDs
  there before generating figures for other models.
- `rdo.py` and `rdo_repind.py` use model-specific chat templates. New model
  families need a tokenizer wrapper and refusal-token configuration.
- Modal runners assume model checkpoints are local directories under
  `/vol/models`. Hugging Face IDs work best for local runs.
