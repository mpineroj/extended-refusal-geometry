#!/usr/bin/env python3
"""
Reproducible analysis script for geometry-of-refusal paper sections.

Sections covered
----------------
  7.1  DIM landscape heatmaps         — 5x3 heatmap grid (ablation/addition/KL)
  7.2  Selected direction quality     — per-model table (layer, pos, scores, DIM-ASR)
  7.3  DIM–RDO geometric alignment    — cosine similarity distributions (K=1 pool)
  7.5  Per-basis causal load          — ablation ASR per cone basis, per K
  7.8  DIM vs RDO attack comparison   — grouped bar chart (DIM / RDO-K1 / RDO-cone)

Usage
-----
    python analyze_geometry_results.py
    python analyze_geometry_results.py --results-root geometry_results_final --output-dir figures
    python analyze_geometry_results.py --sections 72,75,78   # run subset

Outputs (all in --output-dir)
------------------------------
    7.1_dim_heatmap_grid.{png,pdf}
    7.2_dim_quality_table.{csv,txt}
    7.3_dim_rdo_alignment.{png,pdf}
    7.5_causal_load.{png,pdf}
    7.8_dim_vs_rdo.{png,pdf}
    summary.json

Caveats / known limitations
-----------------------------
  7.3  Pool-based cosine distributions: uses all K=1 candidate vectors from
       vectors_<run_id>.pt, not the final selected vector. To get exact
       selected-vector alignment, download selected vectors then re-run:

           modal volume get er-geometry-V2 /results ./geometry_results_final/results

  7.5  PRELIMINARY substring-matching ASR. Explanation-only and justification-only
  7.8  baselines are already 0.74 / 0.82 — these models produce hedging responses
       that don't start with standard refusal prefixes, so substring matching is
       unreliable for them. Re-run after modal/rescore.py --use-strongreject completes
       and point --strongreject-dir at the rescore output directory.
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from plot_style import apply_style
    COLORS = apply_style()
except ImportError:
    warnings.warn("plot_style.py not found; using matplotlib defaults.")
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Model registries ─────────────────────────────────────────────────────────

# Canonical model names used for /models, dim_directions, and rdo artifacts.
MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-Instruct-ER-fullweight",
    "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-justification-only",
]

# Results use the same canonical names.
MODEL_TO_RESULTS = {
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-Instruct-ER-fullweight": "Qwen2.5-3B-Instruct-ER-fullweight",
    "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only": "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only": "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-justification-only": "Qwen2.5-3B-Instruct-ER-fullweight-justification-only",
}

DISPLAY_NAMES = {
    "Qwen2.5-3B-Instruct": "Baseline",
    "Qwen2.5-3B-Instruct-ER-fullweight": "ER-full",
    "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only": "ER-refusal-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only": "ER-expl-only",
    "Qwen2.5-3B-Instruct-ER-fullweight-justification-only": "ER-just-only",
}

# Token labels for DIM position axis
TOKEN_LABELS = {
    -5: "<|im_end|>",
    -4: r"\n",
    -3: "<|im_start|>",
    -2: "assistant",
    -1: r"\n",
}

# (run_id, k_dim, shape_1d)
# shape_1d=True  → saved as (d,), eval label has no _basis suffix
# shape_1d=False → saved as (1, d), eval label uses _basis0
_RUNS = {
    "Qwen2.5-3B-Instruct-ER-fullweight": [
        ("w0maf5em", 1, True),
        ("gmdklqzq", 1, True),
        ("p96yrbxz", 1, True),
        ("b19ru62n", 1, False),
        ("r53uw8us", 2, True),
        ("ash19paq", 3, True),
        ("7yvedmxz", 4, True),
        ("5mfanay3", 5, True),
        # RepInd K=1 runs
        ("6mzddjih", 1, True),
        ("uwbp5kqf", 1, True),
        ("yj08xvr2", 1, True),
    ],
    "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only": [
        ("a25kqz2z", 1, True),
        ("l37s78ml", 1, True),
        ("mfapjvmh", 1, True),
        ("m9wncdt4", 1, False),
        ("phctj5e6", 2, True),
        ("npdua4y8", 3, True),
        ("k0uh2n3y", 4, True),
        ("x83tylwy", 5, True),
        # RepInd K=1 runs
        ("veh3j411", 1, True),
        ("b8b2rtkr", 1, True),
        ("f4e4s40d", 1, True),
        ("dsmwv9l5", 1, True),
        ("ex2g0juq", 1, True),
        ("r8jlw91x", 1, True),
        ("lt656cfr", 1, True),
        ("m4h9sk9g", 1, True),
        ("yyj7ulxz", 1, True),
        ("gtfuz7bg", 1, True),
        ("s4r059mz", 1, True),
        ("0sga31jr", 1, True),
        ("jh1oz535", 1, True),
    ],
    "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only": [
        ("91dpzcaa", 1, True),
        ("eha1bscy", 1, True),
        ("zq7jq08v", 1, True),
        ("pf087ohl", 1, False),
        ("2t9sbaa5", 2, True),
        ("c9cube09", 3, True),
        ("focnhp14", 4, True),
        ("scl7sxhf", 5, True),
        # RepInd K=1 runs
        ("o04gd014", 1, True),
        ("z0uhx42g", 1, True),
        ("a608y2ff", 1, True),
        ("fqpsqtlu", 1, True),
        ("d2d5bu10", 1, True),
        ("6h3xv5jb", 1, True),
        ("vxdscvjd", 1, True),
    ],
    "Qwen2.5-3B-Instruct-ER-fullweight-justification-only": [
        ("3c6f1how", 1, True),
        ("g1r54kwm", 1, True),
        ("lc0a03mc", 1, True),
        ("4bfkv0os", 1, False),
        ("qamtpr39", 2, True),
        ("xpk1nqch", 3, True),
        ("3b1cvjbd", 4, True),
        ("xw6w68dq", 5, True),
        # RepInd K=1 runs
        ("76y3iuh8", 1, True),
        ("nzbf8fv7", 1, True),
        ("caq07u1r", 1, True),
        ("na1fct5v", 1, True),
        ("xrrwvvk8", 1, True),
        ("4dn2c98a", 1, True),
        ("ywxtxe2s", 1, True),
    ],
    "Qwen2.5-3B-Instruct": [
        ("14tm8q4q", 1, True),
        ("5gg10hfm", 2, True),
        # RepInd K=1 runs
        ("cg9ehnpb", 1, True),
        ("s2o6pudq", 1, True),
        ("uq7m6m30", 1, True),
    ],
}

ER_MODELS = [m for m in MODELS if m != "Qwen2.5-3B-Instruct"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def find_eval_summaries(results_root, results_name):
    """Return sorted list of (label, summary_dict) for all selected-eval summaries."""
    base = os.path.join(results_root, "results", "rdo", results_name)
    found = {}
    for pattern in [
        os.path.join(base, "selected", "evals", "evals", "*", "rdo_eval_summary.json"),
        os.path.join(base, "selected", "evals", "*", "rdo_eval_summary.json"),
    ]:
        for path in glob.glob(pattern):
            key = os.path.basename(os.path.dirname(path))
            if key not in found:
                found[key] = path
    result = []
    for label, path in sorted(found.items()):
        with open(path) as f:
            result.append((label, json.load(f)))
    return result


def find_sr_asrs(results_root, results_name):
    """Return {label: StrongREJECT_score} for all selected-eval labels.

    Tries multiple local path patterns produced by modal volume get artifacts.
    Falls back gracefully — missing keys mean no SR score available for that label.
    """
    base = os.path.join(results_root, "results", "rdo", results_name)
    found = {}
    for pattern in [
        os.path.join(base, "selected", "selected", "evals", "*", "completions",
                     "jailbreakbench_ablation_strongreject_evaluations.json"),
        os.path.join(base, "selected", "evals", "evals", "*", "completions",
                     "jailbreakbench_ablation_strongreject_evaluations.json"),
        os.path.join(base, "selected", "evals", "*", "completions",
                     "jailbreakbench_ablation_strongreject_evaluations.json"),
    ]:
        for path in glob.glob(pattern):
            # label = the eval directory name (e.g. "selected_vector_w0maf5em")
            label = os.path.basename(os.path.dirname(os.path.dirname(path)))
            if label not in found:
                try:
                    with open(path) as f:
                        found[label] = json.load(f).get("StrongREJECT_score")
                except Exception:
                    pass
    return found


def _is_k1(label):
    return "basis" not in label


def _basis_idx(label):
    if "basis" not in label:
        return None
    return int(label.split("basis")[-1])


def _run_id_from_label(label):
    part = label.replace("selected_vector_", "")
    return part.split("_basis")[0] if "_basis" in part else part


def _save(fig, output_dir, stem, dpi=220):
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {path}")


# ── Section 7.1 — DIM landscape heatmaps ─────────────────────────────────────

def section_71(results_root, output_dir):
    print("\n[7.1] DIM landscape heatmaps")
    try:
        from plot_dim_comparison_grid import plot_grid
        # Passing an absolute path as output_name: os.path.join(results_root, abs_path) = abs_path
        outname = os.path.join(output_dir, "7.1_dim_heatmap_grid")
        plot_grid(results_root=results_root, output_name=outname, dpi=220)
        return {"status": "ok"}
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return {"status": "error", "msg": str(exc)}


# ── Section 7.2 — Selected direction quality table ────────────────────────────

def section_72(results_root, output_dir):
    print("\n[7.2] Selected direction quality table")

    rows = []
    for model_name in MODELS:
        label = DISPLAY_NAMES[model_name]
        dim_dir = os.path.join(results_root, "results", "dim_directions", model_name)
        meta_path = os.path.join(dim_dir, "direction_metadata.json")
        evals_path = os.path.join(dim_dir, "select_direction", "direction_evaluations.json")
        if not os.path.exists(meta_path) or not os.path.exists(evals_path):
            print(f"  SKIP {label}: missing metadata or evaluations")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        with open(evals_path) as f:
            evals = json.load(f)

        layer = int(meta["layer"])
        pos = int(meta.get("pos", meta.get("position")))
        matched = [e for e in evals if int(e["layer"]) == layer and int(e["position"]) == pos]
        if not matched:
            print(f"  WARN {label}: (layer={layer}, pos={pos}) not found in evaluations")
            continue
        e = matched[0]

        def _read_asr(tag):
            # StrongREJECT preferred; fall back to substring matching
            sr = os.path.join(dim_dir, "completions", f"jailbreakbench_{tag}_strongreject_evaluations.json")
            if os.path.exists(sr):
                with open(sr) as f:
                    return json.load(f).get("StrongREJECT_score")
            sub = os.path.join(dim_dir, "completions", f"jailbreakbench_{tag}_evaluations.json")
            if not os.path.exists(sub):
                return None
            with open(sub) as f:
                return json.load(f).get("substring_matching_success_rate")

        rows.append({
            "model": label,
            "layer": layer,
            "n_layers": 36,
            "pos": pos,
            "pos_token": TOKEN_LABELS.get(pos, str(pos)),
            "refusal_score": round(float(e["refusal_score"]), 3),
            "steering_score": round(float(e["steering_score"]), 3),
            "kl_div": round(float(e["kl_div_score"]), 3),
            "dim_baseline_asr": _read_asr("baseline"),
            "dim_ablation_asr": _read_asr("ablation"),
        })

    if not rows:
        print("  No rows computed.")
        return {}

    def _fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    header = (
        f"{'Model':<22} {'Layer':>6} {'Pos':>4} {'Token':>14}"
        f" {'RefusalSc':>10} {'SteerSc':>8} {'KL':>6} {'BL-ASR':>8} {'Abl-ASR':>8}"
    )
    sep = "-" * len(header)
    print(f"  {header}")
    print(f"  {sep}")
    for r in rows:
        print(
            f"  {r['model']:<22} {r['layer']:>6} {r['pos']:>4} {r['pos_token']:>14}"
            f" {r['refusal_score']:>10.3f} {r['steering_score']:>8.3f} {r['kl_div']:>6.3f}"
            f" {_fmt(r['dim_baseline_asr']):>8} {_fmt(r['dim_ablation_asr']):>8}"
        )

    csv_path = os.path.join(output_dir, "7.2_dim_quality_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(output_dir, "7.2_dim_quality_table.txt")
    with open(txt_path, "w") as f:
        f.write(header + "\n" + sep + "\n")
        for r in rows:
            f.write(
                f"{r['model']:<22} {r['layer']:>6} {r['pos']:>4} {r['pos_token']:>14}"
                f" {r['refusal_score']:>10.3f} {r['steering_score']:>8.3f} {r['kl_div']:>6.3f}"
                f" {_fmt(r['dim_baseline_asr']):>8} {_fmt(r['dim_ablation_asr']):>8}\n"
            )

    print(f"  -> {csv_path}")
    print(f"  -> {txt_path}")
    return {"rows": rows}


# ── Section 7.3 — DIM–RDO geometric alignment ────────────────────────────────
#
# Tries exact alignment (selected_vector_*.pt) first; falls back to pool-based
# distribution if selected vectors have not been downloaded yet.
#
# To get exact alignment, download selected vectors from Modal:
#   for model in Qwen2.5-3B-Instruct-ER-fullweight \
#                Qwen2.5-3B-Instruct-ER-fullweight-refusal-only \
#                Qwen2.5-3B-Instruct-ER-fullweight-explanation-only \
#                Qwen2.5-3B-Instruct-ER-fullweight-justification-only; do
#     mkdir -p geometry_results_final/results/rdo/${model}/selected
#     modal volume get er-geometry-V2 /results/rdo/${model}/selected/ \
#       geometry_results_final/results/rdo/${model}/selected/ --force
#   done

def _load_vec(path):
    """Load a direction .pt and return a normalised 1-D float32 tensor."""
    import torch
    import torch.nn.functional as F
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = torch.load(path, map_location="cpu", weights_only=True)
    return F.normalize(v.float().flatten(), dim=0)


def section_73(results_root, output_dir):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("\n[7.3] SKIP: torch not available in this environment.")
        return {"status": "skip", "reason": "torch not found"}

    print("\n[7.3] DIM–RDO geometric alignment")

    results = {}

    for model_name in ER_MODELS:
        results_name = MODEL_TO_RESULTS[model_name]
        label = DISPLAY_NAMES[model_name]

        dim_pt = os.path.join(results_root, "results", "dim_directions", model_name, "direction.pt")
        if not os.path.exists(dim_pt):
            print(f"  SKIP {label}: DIM direction.pt not found")
            continue

        dim_dir = _load_vec(dim_pt)

        # ── Try exact selected vectors first ──────────────────────────────────
        exact_cosines = {}   # run_id → float
        for run_id, k, _ in _RUNS.get(results_name, []):
            if k != 1:
                continue
            sel_pt = os.path.join(
                results_root, "results", "rdo", results_name,
                "selected", "selected", f"selected_vector_{run_id}.pt"
            )
            if os.path.exists(sel_pt):
                v = _load_vec(sel_pt)
                exact_cosines[run_id] = float(torch.dot(dim_dir, v))

        if exact_cosines:
            mode = "exact"
            cosines = list(exact_cosines.values())
            print(
                f"  {label} [exact, n={len(cosines)}]: "
                + ", ".join(f"{rid}={c:.3f}" for rid, c in exact_cosines.items())
            )

        # ── Fall back to pool-based distribution ──────────────────────────────
        else:
            mode = "pool"
            cosines = []
            for run_id, k, _ in _RUNS.get(results_name, []):
                if k != 1:
                    continue
                pool_pt = os.path.join(
                    results_root, "results", "rdo", model_name,
                    "vectors", f"vectors_{run_id}.pt"
                )
                if not os.path.exists(pool_pt):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pool = torch.load(pool_pt, map_location="cpu", weights_only=True)
                for candidate in pool:
                    v = F.normalize(candidate.float().flatten(), dim=0)
                    cosines.append(float(torch.dot(dim_dir, v)))

            if not cosines:
                print(f"  SKIP {label}: no selected vectors or pool files found")
                continue
            print(
                f"  {label} [pool, n={len(cosines)}]: "
                f"mean={np.mean(cosines):.3f}, max={np.max(cosines):.3f}, "
                f"std={np.std(cosines):.3f}"
            )

        results[model_name] = {
            "label": label,
            "mode": mode,
            "cosines": cosines,
            "run_cosines": exact_cosines if mode == "exact" else {},
            "n": len(cosines),
            "mean": float(np.mean(cosines)),
            "max": float(np.max(cosines)),
            "min": float(np.min(cosines)),
            "std": float(np.std(cosines)),
        }

    if not results:
        return {}

    ordered = [m for m in ER_MODELS if m in results]
    exact_mode = all(results[m]["mode"] == "exact" for m in ordered)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    positions = list(range(len(ordered)))

    if exact_mode:
        # Bar chart: one bar per model (mean of selected K=1 runs), dots for each run
        means = [results[m]["mean"] for m in ordered]
        colors = [COLORS[i % len(COLORS)] for i in range(len(ordered))]
        ax.bar(positions, means, color=colors, alpha=0.75, zorder=2)

        rng = np.random.default_rng(42)
        for i, m in enumerate(ordered):
            cosines = results[m]["cosines"]
            jitter = rng.uniform(-0.12, 0.12, len(cosines))
            ax.scatter(
                np.full(len(cosines), i) + jitter, cosines,
                s=40, color="black", zorder=3, alpha=0.8
            )
        ax.set_ylabel("Cosine similarity (DIM vs. selected RDO K=1 vector)")
        ax.set_title("DIM–RDO Geometric Alignment (exact selected vectors)",
                     fontsize=12, fontweight="bold")
        note = "Each dot = one selected K=1 RDO run. Bar = mean across runs per model."
    else:
        # Violin plot for pool-based
        data = [results[m]["cosines"] for m in ordered]
        parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(COLORS[i % len(COLORS)])
            pc.set_alpha(0.65)
        rng = np.random.default_rng(42)
        for i, (d, pos) in enumerate(zip(data, positions)):
            ax.scatter(
                np.full(len(d), pos) + rng.uniform(-0.07, 0.07, len(d)), d,
                s=9, alpha=0.35, color=COLORS[i % len(COLORS)], zorder=3,
            )
        ax.set_ylabel("Cosine similarity (DIM vs. RDO K=1 candidates)")
        ax.set_title("DIM–RDO Geometric Alignment (pool-based — download selected vectors for exact)",
                     fontsize=11, fontweight="bold")
        note = (
            "Pool-based: distribution over all K=1 RDO candidates, not final selected vectors. "
            "See script header for download command."
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([results[m]["label"] for m in ordered], fontsize=10)
    ax.set_ylim(-0.05, max(0.6, max(results[m]["max"] for m in ordered) + 0.1))

    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="dimgray", style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, output_dir, "7.3_dim_rdo_alignment")
    plt.close(fig)

    return {m: {k: v for k, v in r.items() if k != "cosines"} for m, r in results.items()}


# ── Section 7.4 — Cone dimensionality across variants ────────────────────────

def section_74(results_root, output_dir, chance_threshold=0.1):
    print("\n[7.4] Cone dimensionality across variants")

    model_data = {}

    for model_name in MODELS:
        results_name = MODEL_TO_RESULTS[model_name]
        label = DISPLAY_NAMES[model_name]
        cone_runs = [(rid, k) for rid, k, _ in _RUNS.get(results_name, []) if k > 1]

        if not cone_runs:
            print(f"  SKIP {label}: no cone runs in _RUNS")
            continue

        k_asrs = {}
        for run_id, k in cone_runs:
            summary_path = os.path.join(
                results_root, "results", "rdo", results_name,
                "cone_samples", run_id, "summary.json",
            )
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                s = json.load(f)
            asrs = s.get("all_asrs", [])
            if not asrs:
                continue
            # Keep the run with higher median if multiple runs share the same K
            if k not in k_asrs or float(np.median(asrs)) > float(np.median(k_asrs[k])):
                k_asrs[k] = asrs

        if not k_asrs:
            print(f"  SKIP {label}: no cone_samples data (run modal/cone_samples.py first)")
            continue

        passing = [k for k, asrs in k_asrs.items() if float(np.median(asrs)) > chance_threshold]
        max_k = max(passing) if passing else 0
        model_data[model_name] = {"label": label, "k_asrs": k_asrs, "max_k": max_k}

        for k, asrs in sorted(k_asrs.items()):
            print(
                f"  {label} K={k}: median={float(np.median(asrs)):.3f} "
                f"mean={float(np.mean(asrs)):.3f} n={len(asrs)}"
            )
        print(f"  {label}: K_max={max_k}")

    if not model_data:
        print("  No results. Run modal/cone_samples.py first.")
        return {}

    ordered = [m for m in MODELS if m in model_data]
    ncols = min(len(ordered), 3)
    nrows = (len(ordered) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax_idx, model_name in enumerate(ordered):
        ax = axes_flat[ax_idx]
        d = model_data[model_name]
        k_vals = sorted(d["k_asrs"].keys())
        data = [d["k_asrs"][k] for k in k_vals]

        bp = ax.boxplot(
            data, positions=k_vals, widths=0.5, patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.75)

        ax.axhline(
            chance_threshold, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
            label=f"Chance threshold ({chance_threshold})",
        )
        ax.set_xlabel("Cone dimension K", fontsize=10)
        ax.set_ylabel("Ablation ASR (substring matching)", fontsize=9)
        ax.set_title(f"{d['label']}   (K_max={d['max_k']})", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.02, 1.08)
        ax.set_xticks(k_vals)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes_flat[len(ordered):]:
        ax.set_visible(False)

    fig.suptitle(
        "Cone Dimensionality of Refusal (Section 7.4)\n"
        f"Largest K with median ablation ASR > {chance_threshold}",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        f"Each box = distribution over 256 sampled cone directions "
        f"(positive orthant of K-dim subspace). "
        f"Red dashed line = chance threshold ({chance_threshold}).",
        ha="center", fontsize=8, color="dimgray", style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, output_dir, "7.4_cone_dimensionality")
    plt.close(fig)

    return {
        m: {
            "label": model_data[m]["label"],
            "max_k": model_data[m]["max_k"],
            "k_medians": {k: float(np.median(asrs)) for k, asrs in model_data[m]["k_asrs"].items()},
        }
        for m in ordered
    }


# ── Section 7.5 — Per-basis causal load ──────────────────────────────────────

def section_75(results_root, output_dir):
    print("\n[7.5] Per-basis causal load")

    model_data = {}
    any_sr_used = False

    for model_name in ER_MODELS:
        results_name = MODEL_TO_RESULTS[model_name]
        label = DISPLAY_NAMES[model_name]
        summaries = find_eval_summaries(results_root, results_name)
        sr_asrs = find_sr_asrs(results_root, results_name)

        # Collect per-run, per-basis ablation ASR for K > 1 runs only
        run_bases = {}
        for lbl, s in summaries:
            if _is_k1(lbl):
                continue
            rid = _run_id_from_label(lbl)
            bidx = _basis_idx(lbl)
            # StrongREJECT preferred; fall back to substring matching from summary
            if lbl in sr_asrs and sr_asrs[lbl] is not None:
                ab = sr_asrs[lbl]
                any_sr_used = True
            else:
                ab = s.get("results", {}).get("ablation")
            if ab is None or bidx is None:
                continue
            run_bases.setdefault(rid, {})[bidx] = float(ab)

        # Group by K; keep best run per K (by max basis ASR)
        best_per_k = {}
        for rid, bases in run_bases.items():
            k = max(bases.keys()) + 1
            ordered = [bases.get(i, float("nan")) for i in range(k)]
            cur_best = best_per_k.get(k)
            if cur_best is None or max(ordered) > max(cur_best[1]):
                best_per_k[k] = (rid, ordered)

        model_data[model_name] = {"label": label, "best_per_k": best_per_k}

        for k, (rid, asrs) in sorted(best_per_k.items()):
            print(f"  {label} K={k} ({rid}): {[round(a, 3) for a in asrs]}")

    # 2×2 figure, one panel per ER model
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.flatten()

    for ax_idx, model_name in enumerate(ER_MODELS):
        ax = axes_flat[ax_idx]
        label = model_data[model_name]["label"]
        best_per_k = model_data[model_name]["best_per_k"]
        k_values = sorted(best_per_k.keys())

        if not k_values:
            ax.set_visible(False)
            continue

        max_bases = max(len(best_per_k[k][1]) for k in k_values)
        bw = 0.72 / max(len(k_values), 1)
        x = np.arange(max_bases)

        for ki, k in enumerate(k_values):
            rid, asrs = best_per_k[k]
            offset = (ki - (len(k_values) - 1) / 2) * bw
            bar_x = np.arange(len(asrs))
            ax.bar(
                bar_x + offset, asrs,
                width=bw * 0.88,
                label=f"K={k}",
                color=COLORS[ki % len(COLORS)],
                alpha=0.85,
            )

        ax.set_xticks(x[:max_bases])
        ax.set_xticklabels([f"basis {i}" for i in range(max_bases)], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Ablation ASR", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

    fig.suptitle(
        "Per-Basis Causal Load in Refusal Cones\n(best run per K, by max basis ASR)",
        fontsize=12, fontweight="bold",
    )
    scorer_note = "StrongREJECT" if any_sr_used else "substring matching (preliminary)"
    fig.text(
        0.5, 0.01,
        f"ASR scorer: {scorer_note}. Best run per K selected by max basis ASR.",
        ha="center", fontsize=8, color="dimgray", style="italic",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    _save(fig, output_dir, "7.5_causal_load")
    plt.close(fig)

    return {
        m: {k: {"run_id": v[0], "basis_asrs": v[1]}
            for k, v in model_data[m]["best_per_k"].items()}
        for m in ER_MODELS
    }


# ── Section 7.8 — DIM vs RDO attack comparison ───────────────────────────────
#
# NOTE: Substring matching — preliminary. See module docstring.

def section_78(results_root, output_dir):
    print("\n[7.8] DIM vs RDO attack comparison")

    def _read_dim_asr(model_name, tag):
        comp_dir = os.path.join(results_root, "results", "dim_directions", model_name, "completions")
        sr = os.path.join(comp_dir, f"jailbreakbench_{tag}_strongreject_evaluations.json")
        if os.path.exists(sr):
            with open(sr) as f:
                return json.load(f).get("StrongREJECT_score")
        sub = os.path.join(comp_dir, f"jailbreakbench_{tag}_evaluations.json")
        if not os.path.exists(sub):
            return None
        with open(sub) as f:
            return json.load(f).get("substring_matching_success_rate")

    results = {}
    any_sr_used = False

    for model_name in ER_MODELS:
        results_name = MODEL_TO_RESULTS[model_name]
        label = DISPLAY_NAMES[model_name]
        summaries = find_eval_summaries(results_root, results_name)
        sr_asrs = find_sr_asrs(results_root, results_name)
        if sr_asrs:
            any_sr_used = True

        best_k1_abl, best_k1_lbl = None, None
        best_cone_abl, best_cone_lbl = None, None

        for lbl, s in summaries:
            # StrongREJECT preferred; fall back to substring from rdo_eval_summary
            if lbl in sr_asrs and sr_asrs[lbl] is not None:
                ab = sr_asrs[lbl]
            else:
                ab = s.get("results", {}).get("ablation")
            if ab is None:
                continue
            ab = float(ab)
            if _is_k1(lbl):
                if best_k1_abl is None or ab > best_k1_abl:
                    best_k1_abl, best_k1_lbl = ab, lbl
            else:
                if best_cone_abl is None or ab > best_cone_abl:
                    best_cone_abl, best_cone_lbl = ab, lbl

        results[model_name] = {
            "label": label,
            "baseline_asr": _read_dim_asr(model_name, "baseline"),
            "dim_ablation_asr": _read_dim_asr(model_name, "ablation"),
            "rdo_k1_best_asr": best_k1_abl,
            "rdo_k1_best_run": best_k1_lbl,
            "rdo_cone_best_asr": best_cone_abl,
            "rdo_cone_best_run": best_cone_lbl,
        }
        r = results[model_name]
        print(
            f"  {label}: baseline={r['baseline_asr']}, DIM-abl={r['dim_ablation_asr']},"
            f" RDO-K1={r['rdo_k1_best_asr']} ({r['rdo_k1_best_run']}),"
            f" RDO-cone={r['rdo_cone_best_asr']} ({r['rdo_cone_best_run']})"
        )

    ordered = [m for m in ER_MODELS if m in results]
    labels = [results[m]["label"] for m in ordered]
    n = len(labels)
    x = np.arange(n)
    bw = 0.19

    scorer_label = "StrongREJECT" if any_sr_used else "substring matching (preliminary)"
    fig, ax = plt.subplots(figsize=(9, 5))

    bar_groups = [
        ([results[m]["baseline_asr"] or 0 for m in ordered], "Baseline (no attack)", COLORS[4], -1.5 * bw),
        ([results[m]["dim_ablation_asr"] or 0 for m in ordered], "DIM ablation", COLORS[0], -0.5 * bw),
        ([results[m]["rdo_k1_best_asr"] or 0 for m in ordered], "RDO K=1 (best)", COLORS[1], 0.5 * bw),
        ([results[m]["rdo_cone_best_asr"] or 0 for m in ordered], "RDO cone (best basis)", COLORS[2], 1.5 * bw),
    ]

    for vals, lbl, color, off in bar_groups:
        ax.bar(x + off, vals, width=bw * 0.9, label=lbl, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(f"Attack success rate (ASR)\n[{scorer_label}]")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("DIM vs RDO Attack Success Rate per ER Model", fontsize=12, fontweight="bold")

    fig.text(
        0.5, 0.01,
        f"ASR scorer: {scorer_label}. Best K=1 and cone runs selected by highest ablation ASR.",
        ha="center", fontsize=8, color="dimgray", style="italic",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, output_dir, "7.8_dim_vs_rdo")
    plt.close(fig)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-root",
        default=os.path.join(SCRIPT_DIR, "..", "geometry_results_final"),
        help="Path to geometry_results_final. Default: <repo>/geometry_results_final",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "..", "figures"),
        help="Output directory for figures and tables. Default: <repo>/figures",
    )
    p.add_argument(
        "--sections",
        default="71,72,73,74,75,78",
        help="Comma-separated section IDs to run. Default: all (71,72,73,74,75,78)",
    )
    p.add_argument(
        "--cone-threshold",
        type=float,
        default=0.1,
        help="Chance threshold for section 7.4 (default: 0.1)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sections = set(args.sections.split(","))

    print(f"Results root : {results_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Sections     : {', '.join(sorted(sections))}")

    dispatch = {
        "71": section_71,
        "72": section_72,
        "73": section_73,
        "74": lambda r, o: section_74(r, o, chance_threshold=args.cone_threshold),
        "75": section_75,
        "78": section_78,
    }
    summary = {}
    for sid in ["71", "72", "73", "74", "75", "78"]:
        if sid in sections:
            summary[f"7.{sid[1:]}"] = dispatch[sid](results_root, output_dir)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
