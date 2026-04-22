"""
Compute Spearman correlation between perturbation strength and per-perturbation
model performance for each (model, metric) pair.

Perturbation strength can be measured by either:
  - E-distance from control (--strength edist, requires --edist-matrix)
  - Number of DEGs (--strength ndeg, requires --quality-cache)

For each model and metric, the script correlates the strength measure with
that perturbation's metric value across all perturbations. This tests whether
metrics behave monotonically with perturbation strength.

With --plot, generates a 2x2 scatter-plot PNG per metric showing strength vs.
metric value for four key models: interpolated duplicate, dataset mean,
PRESAGE, and technical duplicate.

Usage:
    python scripts/edist_metric_correlation.py \
        --detailed-metrics outputs/.../detailed_metrics.csv \
        --strength edist --edist-matrix wessels23_edist_matrix.pkl \
        [-o output.csv] [--plot]

    python scripts/edist_metric_correlation.py \
        --detailed-metrics outputs/.../detailed_metrics.csv \
        --strength ndeg --quality-cache analyses/calibration/results/dataset_quality_cache.pkl \
        --dataset wessels23 \
        [-o output.csv] [--plot]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

matplotlib.rcParams["font.family"] = "sans-serif"

PLOT_MODELS = [
    ("technical_duplicate", "Technical Duplicate"),
    ("interpolated_duplicate", "Interpolated Duplicate"),
    ("dataset_mean", "Dataset Mean"),
]

LOWER_IS_BETTER = {"mse", "mse_degs", "mae", "mae_degs", "wmse", "wmae"}


def infer_dataset_prefix(edist_path: str) -> str:
    """Infer the dataset prefix from the edist matrix filename.

    E.g. 'wessels23_edist_matrix.pkl' -> 'wessels23'
    """
    stem = Path(edist_path).stem  # e.g. 'wessels23_edist_matrix'
    suffix = "_edist_matrix"
    if not stem.endswith(suffix):
        raise ValueError(
            f"Cannot infer dataset prefix from '{edist_path}'. "
            f"Expected filename ending with '{suffix}.pkl'"
        )
    return stem[: -len(suffix)]


def build_perturbation_mapping(
    detailed_perturbations: list[str],
    strength_keys: set[str],
    dataset_prefix: str,
) -> dict[str, str]:
    """Build a mapping from detailed_metrics perturbation names to strength-measure keys.

    Strategy:
    1. Strip '{dataset_prefix}_' from detailed_metrics perturbation names
    2. Try direct lookup in strength_keys
    3. If no match, try replacing '+' with '_' (handles wessels23 edist combo naming)
    """
    prefix = f"{dataset_prefix}_"
    mapping = {}

    for pert in detailed_perturbations:
        if not pert.startswith(prefix):
            continue
        stripped = pert[len(prefix):]

        if stripped in strength_keys:
            mapping[pert] = stripped
        elif stripped.replace("+", "_") in strength_keys:
            mapping[pert] = stripped.replace("+", "_")

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Spearman correlation between perturbation E-distance and metric performance"
    )
    parser.add_argument(
        "--detailed-metrics",
        required=True,
        help="Path to detailed_metrics.csv (columns: model, perturbation, metric, value)",
    )
    parser.add_argument(
        "--strength",
        choices=["edist", "ndeg"],
        default="edist",
        help="Perturbation strength measure: 'edist' (E-distance from control) or 'ndeg' (number of DEGs)",
    )
    parser.add_argument(
        "--edist-matrix",
        default=None,
        help="Path to {dataset}_edist_matrix.pkl (required when --strength edist)",
    )
    parser.add_argument(
        "--quality-cache",
        default=None,
        help="Path to dataset_quality_cache.pkl (required when --strength ndeg)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name key in quality cache (required when --strength ndeg; "
             "auto-inferred from --edist-matrix when --strength edist)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: derived from detailed-metrics path)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate 2x2 scatter plots (strength vs metric) per metric for key models",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for scatter plot PNGs (default: {strength}_plots_{dataset}/ next to output CSV)",
    )
    args = parser.parse_args()

    # --- Resolve strength measure and dataset prefix ---
    if args.strength == "edist":
        if args.edist_matrix is None:
            parser.error("--edist-matrix is required when --strength edist")
        edist = pd.read_pickle(args.edist_matrix)
        dataset_prefix = args.dataset or infer_dataset_prefix(args.edist_matrix)
        strength_label = "E-dist"
        strength_short = "edist"
    elif args.strength == "ndeg":
        if args.quality_cache is None:
            parser.error("--quality-cache is required when --strength ndeg")
        if args.dataset is None:
            parser.error("--dataset is required when --strength ndeg")
        with open(args.quality_cache, "rb") as f:
            quality_cache = pickle.load(f)
        dataset_prefix = args.dataset
        strength_label = "Number of DEGs"
        strength_short = "ndeg"

    # Load detailed metrics
    dm = pd.read_csv(args.detailed_metrics)
    all_perturbations = dm["perturbation"].unique().tolist()

    # Build strength values per perturbation
    if args.strength == "edist":
        strength_keys = set(edist.index)
        pert_mapping = build_perturbation_mapping(all_perturbations, strength_keys, dataset_prefix)
        pert_strength = {
            dm_name: edist.loc[mapped_name, "control"]
            for dm_name, mapped_name in pert_mapping.items()
        }
    elif args.strength == "ndeg":
        deg_counts = quality_cache[dataset_prefix]["deg_counts"]
        strength_keys = set(deg_counts.keys())
        pert_mapping = build_perturbation_mapping(all_perturbations, strength_keys, dataset_prefix)
        pert_strength = {
            dm_name: deg_counts[mapped_name]
            for dm_name, mapped_name in pert_mapping.items()
        }

    print(f"Dataset: {dataset_prefix}")
    print(f"Strength measure: {strength_label}")
    print(f"Perturbations matched: {len(pert_mapping)} / {len(all_perturbations)}")

    # Get ordered lists of models and metrics (preserving original order)
    models = list(dict.fromkeys(dm["model"]))
    metrics = list(dict.fromkeys(dm["metric"]))

    # Compute Spearman correlation for each (model, metric) pair
    results = {}
    for model in models:
        results[model] = {}
        dm_model = dm[dm["model"] == model]

        for metric in metrics:
            dm_metric = dm_model[dm_model["metric"] == metric]

            # Build aligned arrays of (strength, metric_value) for matched perturbations
            strength_vals = []
            metric_vals = []
            for _, row in dm_metric.iterrows():
                pert = row["perturbation"]
                if pert in pert_strength:
                    strength_vals.append(pert_strength[pert])
                    metric_vals.append(row["value"])

            if len(strength_vals) >= 3:
                rho, _ = spearmanr(strength_vals, metric_vals)
                results[model][metric] = rho
            else:
                results[model][metric] = np.nan

    # Build output DataFrame
    result_df = pd.DataFrame(results).T
    result_df.index.name = "model"
    result_df = result_df[metrics]  # preserve metric order

    # Determine output path
    if args.output is None:
        output_path = str(
            Path(args.detailed_metrics).parent / f"{strength_short}_metric_correlation.csv"
        )
    else:
        output_path = args.output

    result_df.to_csv(output_path)
    print(f"\nSaved correlation table to {output_path}")
    print(f"\nSpearman rho ({strength_label} vs. metric value):\n")
    print(result_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # --- Scatter plots ---
    if args.plot:
        if args.plot_dir is None:
            plot_dir = Path(output_path).parent / f"{strength_short}_plots_{dataset_prefix}"
        else:
            plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Pre-extract per-model data: {model: {perturbation: {metric: value}}}
        model_pert_metric = {}
        for model_key, _ in PLOT_MODELS:
            dm_model = dm[dm["model"] == model_key]
            pert_data = {}
            for _, row in dm_model.iterrows():
                pert = row["perturbation"]
                if pert in pert_strength:
                    pert_data.setdefault(pert, {})[row["metric"]] = row["value"]
            model_pert_metric[model_key] = pert_data

        # Compute global x-axis range across all perturbations (fixed across all plots)
        all_strength_vals = np.array(list(pert_strength.values()), dtype=float)
        positive_vals = all_strength_vals[all_strength_vals > 0]
        x_lo = positive_vals.min() * 0.7
        x_hi = positive_vals.max() * 1.4

        for metric in metrics:
            # First pass: collect all y-values across the four panels to get shared y limits
            panel_data = []
            for model_key, model_label in PLOT_MODELS:
                pert_data = model_pert_metric[model_key]
                x_vals = []
                y_vals = []
                for pert, mdict in pert_data.items():
                    if metric in mdict:
                        x_vals.append(pert_strength[pert])
                        y_vals.append(mdict[metric])
                panel_data.append((model_key, model_label, np.array(x_vals), np.array(y_vals)))

            y_log = metric in LOWER_IS_BETTER

            all_y = np.concatenate([d[3] for d in panel_data if len(d[3]) > 0])
            all_y = all_y[np.isfinite(all_y)]
            if y_log:
                all_y = all_y[all_y > 0]

            if len(all_y) == 0:
                y_lo, y_hi = 0.0, 1.0
            elif y_log:
                y_lo = all_y.min() * 0.7
                y_hi = all_y.max() * 1.4
            else:
                y_margin = (all_y.max() - all_y.min()) * 0.05
                y_lo = all_y.min() - y_margin
                y_hi = all_y.max() + y_margin

            # Second pass: plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            y_label = f"log2({metric})" if y_log else metric
            fig.suptitle(f"{dataset_prefix} — {strength_label} vs. {y_label}", fontsize=15, y=0.98)

            for ax, (model_key, model_label, x_vals, y_vals) in zip(axes.flat, panel_data):
                ax.scatter(x_vals, y_vals, s=30, alpha=0.5, edgecolors="none")

                # Annotate with Spearman rho
                if len(x_vals) >= 3:
                    rho, pval = spearmanr(x_vals, y_vals)
                    ax.set_title(f"{model_label}  (ρ = {rho:.3f})", fontsize=13)
                else:
                    ax.set_title(f"{model_label}  (n < 3)", fontsize=13)

                ax.set_xscale("log", base=2)
                ax.set_xlim(x_lo, x_hi)
                if y_log:
                    ax.set_yscale("log", base=2)
                ax.set_ylim(y_lo, y_hi)
                ax.set_xlabel(strength_label, fontsize=12)
                ax.set_ylabel(y_label, fontsize=12)
                ax.tick_params(labelsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig_path = plot_dir / f"{metric}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"\nSaved {len(metrics)} scatter-plot PNGs to {plot_dir}/")


if __name__ == "__main__":
    main()
