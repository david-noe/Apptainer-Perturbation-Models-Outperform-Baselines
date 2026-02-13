"""
Plot an all-by-all Spearman correlation heatmap of *_mean metrics
from a summary_stats.csv file.

Usage:
    python scripts/metric_correlation_heatmap.py <summary_stats.csv> [--output <output.png>]
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'


def main():
    parser = argparse.ArgumentParser(description="Metric correlation heatmap from summary_stats.csv")
    parser.add_argument("input_csv", help="Path to summary_stats.csv")
    parser.add_argument("--output", "-o", default=None, help="Output path for the heatmap image (default: saved next to input)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.set_index("model")

    # Select only *_mean columns and drop any that are all-NaN
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    df_means = df[mean_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    # Strip the _mean suffix for cleaner labels
    clean_names = [c.replace("_mean", "") for c in df_means.columns]
    df_means.columns = clean_names

    # Drop rows (models) that have any NaN in the remaining metrics
    df_means = df_means.dropna(axis=0)

    # Negate error-based metrics so that "better" is always "higher" across all metrics
    # This ensures Spearman correlations reflect agreement in model ranking
    LOWER_IS_BETTER = {"mse", "mse_degs", "mae", "mae_degs", "wmse", "wmae"}
    for col in df_means.columns:
        if col in LOWER_IS_BETTER:
            df_means[col] = -df_means[col]

    # Compute Spearman correlation matrix
    n_metrics = len(df_means.columns)
    corr_matrix = np.ones((n_metrics, n_metrics))
    pval_matrix = np.ones((n_metrics, n_metrics))

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            rho, pval = spearmanr(df_means.iloc[:, i], df_means.iloc[:, j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval

    corr_df = pd.DataFrame(corr_matrix, index=clean_names, columns=clean_names)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 0.7), max(8, n_metrics * 0.6)))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_metrics))
    ax.set_yticks(range(n_metrics))
    ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(clean_names, fontsize=8)

    # Annotate cells with correlation values
    for i in range(n_metrics):
        for j in range(n_metrics):
            val = corr_df.values[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman correlation", fontsize=10)

    ax.set_title("Metric Correlation (Spearman, *_mean values)", fontsize=12, pad=12)
    plt.tight_layout()

    if args.output is None:
        output_path = args.input_csv.replace(".csv", "_metric_correlation_heatmap.png")
    else:
        output_path = args.output

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    plt.close()

    # Print average correlation for each metric against curated reference sets
    REFERENCE_SET_A = {
        "wmse",
        "weighted_r2_deltactrl",
        "weighted_r2_deltapert",
        "nir",
        "pearson_deltactrl_degs",
        "pearson_deltapert_degs",
    }
    REFERENCE_SET_B = {
        "mae",
        "mae_degs",
        "r2_deltactrl",
    }

    ref_a_present = [m for m in REFERENCE_SET_A if m in corr_df.columns]
    ref_b_present = [m for m in REFERENCE_SET_B if m in corr_df.columns]

    if ref_a_present or ref_b_present:
        header_a = "Avg corr (weighted/DEG)"
        header_b = "Avg corr (naive)"
        print(f"\n{'Metric':>35s}   {header_a:>24s}   {header_b:>24s}")
        print(f"{'':->35s}   {'':->24s}   {'':->24s}")
        for metric in corr_df.columns:
            parts = []
            for ref_list in [ref_a_present, ref_b_present]:
                ref_others = [r for r in ref_list if r != metric]
                if ref_others:
                    avg_corr = corr_df.loc[metric, ref_others].mean()
                    parts.append(f"{avg_corr:.4f}")
                else:
                    parts.append("N/A")
            print(f"  {metric:>35s}   {parts[0]:>24s}   {parts[1]:>24s}")
    else:
        print("\nNo reference metrics found in columns, skipping average correlation summary.")


if __name__ == "__main__":
    main()
