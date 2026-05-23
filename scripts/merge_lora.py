"""
How to run:

1) Basic (uses current shell user for scratch paths):
    python scripts/merge_lora.py --user "$USER"

2) Explicit username:
    python scripts/merge_lora.py --user mp3687

3) SLURM example line:
    python scripts/merge_lora.py --user "$USER" --device-map cpu

4) Override output/cache locations:
    python scripts/merge_lora.py --user "$USER" \
         --cache-dir "/scratch/network/$USER/.cache/huggingface" \
         --models-dir "/scratch/network/$USER/models"

5) Write directly into the Modal volume layout from inside Modal:
    python scripts/merge_lora.py --models-dir /vol/models --cache-dir /vol/cache
"""

import argparse
import os


VARIANTS = [
    ('CSMaya/er-ablations-qwen2.5-3b', 'full-lora', 'Qwen2.5-3B-Instruct-ER-fullweight'),
    ('CSMaya/er-ablations-qwen2.5-3b', 'justification-only-lora', 'Qwen2.5-3B-Instruct-ER-fullweight-justification-only'),
    ('CSMaya/er-ablations-qwen2.5-3b', 'explanation-only-lora', 'Qwen2.5-3B-Instruct-ER-fullweight-explanation-only'),
    ('CSMaya/er-ablations-qwen2.5-3b', 'refusal-only-lora', 'Qwen2.5-3B-Instruct-ER-fullweight-refusal-only'),
]


def parse_args():
    default_user = os.environ.get('USER', 'mp3687')
    parser = argparse.ArgumentParser(
        description='Merge LoRA adapters into full Qwen model checkpoints.'
    )
    parser.add_argument(
        '--user',
        default=default_user,
        help='Username used to build default scratch paths (default: current USER env).',
    )
    parser.add_argument(
        '--base-model',
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Base model name or path.',
    )
    parser.add_argument(
        '--cache-dir',
        default=None,
        help='Hugging Face cache directory. Defaults to /scratch/network/<user>/.cache/huggingface.',
    )
    parser.add_argument(
        '--models-dir',
        default=None,
        help='Directory where merged models are saved. Defaults to /scratch/network/<user>/models.',
    )
    parser.add_argument(
        '--device-map',
        default='cpu',
        help='Device map for loading the base model (default: cpu).',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = args.cache_dir or os.environ.get(
        'HUGGINGFACE_CACHE_DIR',
        os.environ.get('HF_HOME', f'/scratch/network/{args.user}/.cache/huggingface'),
    )
    models_dir = args.models_dir or os.environ.get(
        'GEOMETRY_MODELS_DIR',
        f'/scratch/network/{args.user}/models',
    )

    os.makedirs(models_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache_dir)

    for adapter_repo, subfolder, save_name in VARIANTS:
        print(f'\n=== Merging {save_name} ===')
        save_path = f'{models_dir}/{save_name}'

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )

        kwargs = {'cache_dir': cache_dir}
        if subfolder:
            kwargs['subfolder'] = subfolder
        model = PeftModel.from_pretrained(base, adapter_repo, **kwargs)
        merged = model.merge_and_unload()

        merged.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        print(f'Saved to {save_path}')

        del base, model, merged
        torch.cuda.empty_cache()

    print('\nAll models merged.')


if __name__ == '__main__':
    main()
