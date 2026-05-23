"""
Download the baseline model and ER model variants.

Models downloaded:
  1. Qwen/Qwen2.5-3B-Instruct  →  Qwen2.5-3B-Instruct
  2. rpawar7156/qwen2.5-3b-er  →  Qwen2.5-3B-Instruct-ER-fullweight
  3. CSMaya/er-ablations-qwen2.5-3b (explanation-only)
  4. CSMaya/er-ablations-qwen2.5-3b (justification-only)
  5. CSMaya/er-ablations-qwen2.5-3b (refusal-only)

Usage:
  1. Run on Adroit or another scratch-backed machine:
       conda activate new-rdo
       python scripts/download_models.py

  2. Write directly into the Modal volume layout from inside Modal:
       python scripts/download_models.py --models-dir /vol/models --cache-dir /vol/cache

All models are saved as full-weight checkpoints with tokenizer included.
The base Qwen2.5-3B-Instruct tokenizer is used for rpawar7156/qwen2.5-3b-er.
"""

import argparse
import os


BASE_TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FULL_ER_REPO = "rpawar7156/qwen2.5-3b-er"
ABLATION_REPO = "CSMaya/er-ablations-qwen2.5-3b"


def parse_args():
    user = os.environ.get("USER", "mp3687")
    default_models_dir = os.environ.get(
        "GEOMETRY_MODELS_DIR",
        f"/scratch/network/{user}/models",
    )
    default_cache_dir = os.environ.get(
        "HUGGINGFACE_CACHE_DIR",
        os.environ.get("HF_HOME", f"/scratch/network/{user}/.cache/huggingface"),
    )

    parser = argparse.ArgumentParser(description="Download baseline and ER model variants.")
    parser.add_argument(
        "--models-dir",
        default=default_models_dir,
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--cache-dir",
        default=default_cache_dir,
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--base-tokenizer",
        default=BASE_TOKENIZER,
        help="Tokenizer to use when an ER repo does not ship one.",
    )
    parser.add_argument(
        "--baseline-model",
        default=BASE_MODEL,
        help="Baseline Hugging Face model ID to save as Qwen2.5-3B-Instruct.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not download the baseline model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_root = args.models_dir
    cache = args.cache_dir

    os.makedirs(save_root, exist_ok=True)

    print(f"Models dir: {save_root}")
    print(f"HF cache:   {cache}")

    # ── 1. Baseline model ───────────────────────────────────────────
    if not args.skip_baseline:
        print("=" * 60)
        print(f"1/5  {args.baseline_model}")
        print("=" * 60)

        save_path = os.path.join(save_root, "Qwen2.5-3B-Instruct")
        if os.path.exists(save_path):
            print(f"  already exists: {save_path}, skipping.\n")
        else:
            print("  downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                args.baseline_model,
                torch_dtype=torch.bfloat16,
                cache_dir=cache,
                low_cpu_mem_usage=True,
            )
            model.save_pretrained(save_path, safe_serialization=True)
            del model
            torch.cuda.empty_cache()

            print("  downloading tokenizer...")
            tok = AutoTokenizer.from_pretrained(args.baseline_model, cache_dir=cache)
            tok.save_pretrained(save_path)
            print(f"  saved -> {save_path}\n")

    # ── 2. Riya's full-weight ER model ──────────────────────────────
    print("=" * 60)
    print(f"2/5  {FULL_ER_REPO}")
    print("=" * 60)

    save_path = os.path.join(save_root, "Qwen2.5-3B-Instruct-ER-fullweight")
    if os.path.exists(save_path):
        print(f"  already exists: {save_path}, skipping.\n")
    else:
        print("  downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            FULL_ER_REPO,
            torch_dtype=torch.bfloat16,
            cache_dir=cache,
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(save_path, safe_serialization=True)
        del model
        torch.cuda.empty_cache()

        print(f"  saving tokenizer from {args.base_tokenizer}...")
        tok = AutoTokenizer.from_pretrained(args.base_tokenizer, cache_dir=cache)
        tok.save_pretrained(save_path)
        print(f"  saved -> {save_path}\n")

    # ── 3-5. Component ablation variants ───────────────────────────
    variants = [
        ("explanation-only-lora", "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only"),
        ("justification-only-lora", "Qwen2.5-3B-Instruct-ER-fullweight-justification-only"),
        ("refusal-only-lora", "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only"),
    ]

    for i, (subdir, name) in enumerate(variants, start=3):
        print("=" * 60)
        print(f"{i}/5  {ABLATION_REPO}/{subdir}")
        print("=" * 60)

        save_path = os.path.join(save_root, name)
        if os.path.exists(save_path):
            print(f"  already exists: {save_path}, skipping.\n")
            continue

        print("  downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            ABLATION_REPO,
            subfolder=subdir,
            torch_dtype=torch.bfloat16,
            cache_dir=cache,
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(save_path, safe_serialization=True)
        del model
        torch.cuda.empty_cache()

        print("  downloading tokenizer...")
        try:
            tok = AutoTokenizer.from_pretrained(ABLATION_REPO, subfolder=subdir, cache_dir=cache)
        except Exception:
            print(f"  tokenizer not in repo, using {args.base_tokenizer}")
            tok = AutoTokenizer.from_pretrained(args.base_tokenizer, cache_dir=cache)
        tok.save_pretrained(save_path)
        print(f"  saved -> {save_path}\n")

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Summary - models in", save_root)
    print("=" * 60)
    for entry in sorted(os.listdir(save_root)):
        full = os.path.join(save_root, entry)
        if os.path.isdir(full):
            n_files = len(os.listdir(full))
            print(f"  {entry}/  ({n_files} files)")
    print("\nAll done.")


if __name__ == "__main__":
    main()
