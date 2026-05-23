#!/usr/bin/env python3
"""
Create a DIM direction-selection comparison grid from saved JSON artifacts.

The output is a 5x3 small-multiples grid:
  rows: baseline Qwen and ER variants
  cols: ablation refusal score, addition refusal score, KL divergence

Each cell is a position x layer heatmap. Color scales are shared within each
metric column so rows are directly comparable.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, TwoSlopeNorm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ROWS = [
    ("Qwen2.5-3B-Instruct", "Qwen baseline\nQwen2.5-3B-Instruct"),
    ("Qwen2.5-3B-Instruct-ER-fullweight", "ER-full\nQwen2.5-3B-Instruct-ER-fullweight"),
    ("Qwen2.5-3B-Instruct-ER-fullweight-explanation-only", "ER-explanation\nQwen2.5-3B-Instruct-ER-fullweight-explanation-only"),
    ("Qwen2.5-3B-Instruct-ER-fullweight-justification-only", "ER-justification\nQwen2.5-3B-Instruct-ER-fullweight-justification-only"),
    ("Qwen2.5-3B-Instruct-ER-fullweight-refusal-only", "ER-refusal\nQwen2.5-3B-Instruct-ER-fullweight-refusal-only"),
]

METRIC_COLS = [
    ("refusal_score", "Ablation", "Ablation refusal score", "RdBu_r", True),
    ("steering_score", "Addition", "Addition refusal score", "RdBu_r", True),
    ("kl_div_score", "KL", "KL divergence", "magma", False),
]

CAPTION = (
    "Ablation: refusal score on harmful prompts under ablation of the direction. "
    "Addition: refusal score on harmless prompts under addition of the direction. "
    "KL: divergence on harmless prompts under ablation. Refusal-score columns "
    "share units but measure different interventions. Each metric column uses "
    "one shared color scale across all rows; the marked cell is the selected "
    "DIM direction from direction_metadata.json."
)

TOKEN_LABELS = {
    -5: r"-5: <|im_end|>",
    -4: r"-4: \n",
    -3: r"-3: <|im_start|>",
    -2: r"-2: assistant",
    -1: r"-1: \n",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a 5x3 DIM comparison grid from direction_evaluations.json files."
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(SCRIPT_DIR, "..", "geometry_results_final"),
        help="Path to geometry_results_final. Default: <repo>/geometry_results_final",
    )
    parser.add_argument(
        "--output-name",
        default="dim_direction_comparison_grid",
        help="Output basename. Writes both .png and .pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG DPI.",
    )
    return parser.parse_args()


def load_model_scores(results_root, model_name):
    model_dir = os.path.join(
        results_root,
        "results",
        "dim_directions",
        model_name,
    )
    json_path = os.path.join(
        model_dir,
        "select_direction",
        "direction_evaluations.json",
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {json_path}")

    with open(json_path) as f:
        rows = json.load(f)

    positions = sorted({int(row["position"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})

    matrices = {}
    for metric_key, _, _, _, _ in METRIC_COLS:
        matrix = [[float("nan") for _ in layers] for _ in positions]
        pos_to_i = {pos: i for i, pos in enumerate(positions)}
        layer_to_i = {layer: i for i, layer in enumerate(layers)}

        for row in rows:
            matrix[pos_to_i[int(row["position"])]][layer_to_i[int(row["layer"])]] = float(
                row[metric_key]
            )
        matrices[metric_key] = matrix

    selected_path = os.path.join(model_dir, "direction_metadata.json")
    selected = None
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            selected_meta = json.load(f)
        selected = {
            "position": int(selected_meta.get("pos", selected_meta.get("position"))),
            "layer": int(selected_meta["layer"]),
        }

    return {
        "json_path": json_path,
        "selected_path": selected_path,
        "selected": selected,
        "positions": positions,
        "layers": layers,
        "matrices": matrices,
    }


def flatten(matrix):
    return [value for row in matrix for value in row if value == value]


def make_norm(all_values, centered):
    if centered:
        limit = max(abs(min(all_values)), abs(max(all_values)))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return Normalize(vmin=0.0, vmax=max(all_values))


def plot_grid(results_root, output_name, dpi):
    loaded = {
        model_name: load_model_scores(results_root, model_name)
        for model_name, _ in MODEL_ROWS
    }

    norms = {}
    for metric_key, col_title, _, _, centered in METRIC_COLS:
        all_values = []
        for model_name, _ in MODEL_ROWS:
            all_values.extend(flatten(loaded[model_name]["matrices"][metric_key]))
        norms[metric_key] = make_norm(all_values, centered=centered)
        print(
            f"{col_title} shared scale across rows: "
            f"{norms[metric_key].vmin:.3f} to {norms[metric_key].vmax:.3f}"
        )

    fig = plt.figure(figsize=(18.5, 11.2))
    grid = fig.add_gridspec(
        nrows=len(MODEL_ROWS),
        ncols=6,
        width_ratios=[1.0, 0.035, 1.0, 0.035, 1.0, 0.035],
        height_ratios=[1, 1, 1, 1, 1],
        left=0.24,
        right=0.985,
        bottom=0.085,
        top=0.88,
        wspace=0.11,
        hspace=0.24,
    )
    axes = [
        [fig.add_subplot(grid[row_idx, col_idx * 2]) for col_idx in range(len(METRIC_COLS))]
        for row_idx in range(len(MODEL_ROWS))
    ]
    colorbar_axes = [
        fig.add_subplot(grid[:, col_idx * 2 + 1])
        for col_idx in range(len(METRIC_COLS))
    ]

    images_by_col = {}
    for row_idx, (model_name, row_label) in enumerate(MODEL_ROWS):
        positions = loaded[model_name]["positions"]
        layers = loaded[model_name]["layers"]
        selected = loaded[model_name]["selected"]

        for col_idx, (metric_key, col_title, ylabel, cmap, _) in enumerate(METRIC_COLS):
            ax = axes[row_idx][col_idx]
            matrix = loaded[model_name]["matrices"][metric_key]

            image = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norms[metric_key],
                origin="lower",
            )
            images_by_col[col_idx] = image

            if row_idx == 0:
                ax.set_title(col_title, fontsize=13, pad=8)
            if row_idx == len(MODEL_ROWS) - 1:
                ax.set_xlabel("Layer source of direction")

            xtick_layers = [layer for layer in layers if layer % 5 == 0 or layer == layers[-1]]
            ax.set_xticks([layers.index(layer) for layer in xtick_layers])
            ax.set_xticklabels([str(layer) for layer in xtick_layers], fontsize=8)

            ax.set_yticks(range(len(positions)))
            if col_idx == 0:
                ax.set_yticklabels(
                    [TOKEN_LABELS.get(pos, str(pos)) for pos in positions],
                    fontsize=8,
                )
            else:
                ax.set_yticklabels([])

            if selected is not None:
                selected_pos = selected["position"]
                selected_layer = selected["layer"]
                if selected_pos in positions and selected_layer in layers:
                    x_idx = layers.index(selected_layer)
                    y_idx = positions.index(selected_pos)
                    ax.add_patch(
                        Rectangle(
                            (x_idx - 0.5, y_idx - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=1.7,
                            zorder=4,
                        )
                    )
                    ax.scatter(
                        [x_idx],
                        [y_idx],
                        marker="*",
                        s=110,
                        facecolor="#00d7ff" if metric_key == "kl_div_score" else "#ffd84d",
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=5,
                    )

            ax.tick_params(length=2)

    for row_idx, (_, row_label) in enumerate(MODEL_ROWS):
        bbox = axes[row_idx][0].get_position()
        y_mid = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.015,
            y_mid,
            row_label,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for col_idx, (_, _, ylabel, _, _) in enumerate(METRIC_COLS):
        colorbar = fig.colorbar(
            images_by_col[col_idx],
            cax=colorbar_axes[col_idx],
            orientation="vertical",
        )
        colorbar.set_label(ylabel)

    fig.suptitle(
        "Extended Refusal Models Shrink the Viable-Direction Landscape that DIM Abliteration Depends On",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.24,
        0.925,
        "Each metric column uses one shared color scale across all rows. "
        "Yellow star and black outline mark the selected DIM direction from direction_metadata.json.",
        ha="left",
        va="center",
        fontsize=9,
    )

    os.makedirs(results_root, exist_ok=True)
    png_path = os.path.join(results_root, f"{output_name}.png")
    pdf_path = os.path.join(results_root, f"{output_name}.pdf")
    caption_path = os.path.join(results_root, f"{output_name}_caption.txt")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    with open(caption_path, "w") as f:
        f.write(CAPTION + "\n")

    print(png_path)
    print(pdf_path)
    print(caption_path)


def main():
    args = parse_args()
    plot_grid(
        results_root=os.path.abspath(args.results_root),
        output_name=args.output_name,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
