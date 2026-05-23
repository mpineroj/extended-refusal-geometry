"""
Standalone figure: Ablation ASR by cone basis vector, comparing baseline vs ER models.
SR-based (StrongREJECT evaluator, JailbreakBench n=100), hardcoded from analysis.

Top panel : Baseline vs ER-full across all cone positions (K=1 DIM, K=2 basis 0/1, K=3 basis 0/1/2)
Bottom panel: All five models at K=1 (DIM) and K=2 (basis 0 / basis 1)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── SR per-basis ablation ASR ─────────────────────────────────────────────────
# "Single direction" = DIM ablation ASR (section 7.2, SR)
# Cone K=N basis i = per-basis ablation ASR (section 7.5, SR)

DIM_ASR = {
    "Baseline":        0.780,
    "ER-full":         0.614,
    "ER-refusal-only": 0.833,
    "ER-expl-only":    0.458,
    "ER-just-only":    0.130,
}

CONE_K2_ASR = {
    "Baseline":        [0.812, 0.639],   # 5gg10hfm
    "ER-full":         [0.608, 0.129],   # r53uw8us
    "ER-refusal-only": [0.051, 0.020],   # phctj5e6
    "ER-expl-only":    [0.165, 0.145],   # 2t9sbaa5
    "ER-just-only":    [0.041, 0.007],   # qamtpr39
}

CONE_K3_ASR = {
    # Baseline K=3 not tested (failed induce threshold)
    "ER-full":         [0.662, 0.165, 0.018],  # ash19paq
}

NATURAL_ASR = {
    "Baseline":        0.134,
    "ER-full":         0.158,
    "ER-refusal-only": 0.040,
    "ER-expl-only":    0.295,
    "ER-just-only":    0.029,
}

COLORS = {
    "Baseline":        "#4878CF",
    "ER-full":         "#D65F5F",
    "ER-refusal-only": "#9B59B6",
    "ER-expl-only":    "#E67E22",
    "ER-just-only":    "#2ECC71",
}

DISPLAY = {
    "Baseline":        "Base Qwen2.5-3B",
    "ER-full":         "ER Full-Weight",
    "ER-refusal-only": "ER-refusal only",
    "ER-expl-only":    "ER-explanation only",
    "ER-just-only":    "ER-justification only",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def grouped_bars(ax, groups, model_keys, bar_w=0.15, gap=0.55):
    """
    groups: list of (label, {model: value or None})
    model_keys: ordered list of model keys
    Returns x positions of group centres.
    """
    n_groups = len(groups)
    n_models = len(model_keys)
    total_w = n_models * bar_w
    centres = np.arange(n_groups) * (total_w + gap)

    for gi, (_, vals) in enumerate(groups):
        for mi, mk in enumerate(model_keys):
            v = vals.get(mk)
            if v is None:
                continue
            x = centres[gi] + (mi - (n_models - 1) / 2) * bar_w
            ax.bar(x, v, width=bar_w * 0.9, color=COLORS[mk], alpha=0.88, zorder=3)

    return centres


def add_group_labels(ax, centres, groups, subtitle_y=-0.13, subtitle2_y=-0.21):
    """Add main label + italic subtitle below each group."""
    for ci, (lbl, _) in enumerate(centres):
        parts = lbl.split("\n")
        ax.text(centres[ci], subtitle_y, parts[0], ha="center", va="top",
                fontsize=9, transform=ax.get_xaxis_transform())
        if len(parts) > 1:
            ax.text(centres[ci], subtitle2_y, parts[1], ha="center", va="top",
                    fontsize=7.5, style="italic", color="dimgray",
                    transform=ax.get_xaxis_transform())


def span_brace(ax, centres, idxs, label, y=-0.30):
    """Draw a horizontal brace spanning a set of group centres."""
    x0, x1 = centres[idxs[0]], centres[idxs[-1]]
    ax.annotate("", xy=(x1 + 0.18, y), xytext=(x0 - 0.18, y),
                arrowprops=dict(arrowstyle="-", color="gray", lw=1.0),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"))
    ax.text((x0 + x1) / 2, y - 0.03, label, ha="center", va="top",
            fontsize=7.5, color="dimgray", style="italic",
            transform=ax.get_xaxis_transform())


# ── Figure 1: Baseline vs ER-full ─────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(9, 4.5))

models_top = ["Baseline", "ER-full"]

groups_top = [
    ("DIM\n", {
        "Baseline": DIM_ASR["Baseline"],
        "ER-full":  DIM_ASR["ER-full"],
    }),
    ("Cone 2\nbasis 0", {
        "Baseline": CONE_K2_ASR["Baseline"][0],
        "ER-full":  CONE_K2_ASR["ER-full"][0],
    }),
    ("Cone 2\nbasis 1", {
        "Baseline": CONE_K2_ASR["Baseline"][1],
        "ER-full":  CONE_K2_ASR["ER-full"][1],
    }),
    ("Cone 3\nbasis 0", {
        "Baseline": None,                        # not tested
        "ER-full":  CONE_K3_ASR["ER-full"][0],
    }),
    ("Cone 3\nbasis 1", {
        "Baseline": None,
        "ER-full":  CONE_K3_ASR["ER-full"][1],
    }),
    ("Cone 3\nbasis 2", {
        "Baseline": None,
        "ER-full":  CONE_K3_ASR["ER-full"][2],
    }),
]

centres_top = grouped_bars(ax1, groups_top, models_top, bar_w=0.22, gap=0.45)

# Natural ASR reference lines
for mk, ls in [("Baseline", "--"), ("ER-full", ":")]:
    ax1.axhline(NATURAL_ASR[mk], color=COLORS[mk], linestyle=ls,
                linewidth=1.0, alpha=0.55,
                label=f"Natural ASR — {DISPLAY[mk]} ({NATURAL_ASR[mk]:.2f})")

ax1.set_xticks(centres_top)
ax1.set_xticklabels([g[0].replace("\n", " ") for g in groups_top], fontsize=9)
ax1.set_ylabel("Ablation ASR (StrongREJECT)", fontsize=10)
ax1.set_ylim(0, 1.08)
ax1.set_title("Ablation ASR by Cone Basis — Baseline vs ER-full", fontsize=12, fontweight="bold")
ax1.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

patches = [mpatches.Patch(color=COLORS[m], label=DISPLAY[m]) for m in models_top]
ax1.legend(handles=patches, fontsize=9, loc="upper right")

# Brace annotations
brace_y = -0.28
span_brace(ax1, centres_top, [0], "Single Direction", y=brace_y)
span_brace(ax1, centres_top, [1, 2], "Cone Dim 2", y=brace_y)
span_brace(ax1, centres_top, [3, 4, 5], "Cone Dim 3", y=brace_y)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig1.savefig("figures/cone_basis_baseline_vs_erfull.png", dpi=150, bbox_inches="tight")
fig1.savefig("figures/cone_basis_baseline_vs_erfull.pdf", bbox_inches="tight")
print("Saved: cone_basis_baseline_vs_erfull.{png,pdf}")
plt.close(fig1)


# ── Figure 2: All models at K=1 (DIM) and K=2 ────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 4.5))

all_models = ["Baseline", "ER-full", "ER-refusal-only", "ER-expl-only", "ER-just-only"]

groups_bot = [
    ("DIM\n", {m: DIM_ASR[m] for m in all_models}),
    ("Cone 2\nbasis 0", {m: CONE_K2_ASR[m][0] for m in all_models}),
    ("Cone 2\nbasis 1", {m: CONE_K2_ASR[m][1] for m in all_models}),
]

centres_bot = grouped_bars(ax2, groups_bot, all_models, bar_w=0.13, gap=0.55)

# Single natural ASR line for baseline only (representative)
ax2.axhline(NATURAL_ASR["Baseline"], color=COLORS["Baseline"], linestyle="--",
            linewidth=1.0, alpha=0.5,
            label=f"Baseline natural ASR ({NATURAL_ASR['Baseline']:.2f})")

ax2.set_xticks(centres_bot)
ax2.set_xticklabels([g[0].replace("\n", " ") for g in groups_bot], fontsize=9)
ax2.set_ylabel("Ablation ASR (StrongREJECT)", fontsize=10)
ax2.set_ylim(0, 1.08)
ax2.set_title("Ablation ASR by Cone Basis — All ER Variants", fontsize=12, fontweight="bold")
ax2.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

patches2 = [mpatches.Patch(color=COLORS[m], label=DISPLAY[m]) for m in all_models]
ax2.legend(handles=patches2, fontsize=8, loc="upper right", ncol=2)

span_brace(ax2, centres_bot, [0], "Single Direction", y=brace_y)
span_brace(ax2, centres_bot, [1, 2], "Cone Dim 2", y=brace_y)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig2.savefig("figures/cone_basis_all_models.png", dpi=150, bbox_inches="tight")
fig2.savefig("figures/cone_basis_all_models.pdf", bbox_inches="tight")
print("Saved: cone_basis_all_models.{png,pdf}")
plt.close(fig2)
