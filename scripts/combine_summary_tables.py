#!/usr/bin/env python3
"""
Combine multiple multimodel summary CSV tables into a single LaTeX table.

Takes CSV files produced by plot_multimodel_summary.py (e.g. multimodel_summary_table_nondeg.csv)
and merges them into one LaTeX table with dataset grouping.

Usage:
    python scripts/combine_summary_tables.py \
        outputs/.../additional_results/multimodel_summary_table_nondeg.csv \
        outputs/.../additional_results/multimodel_summary_table_nondeg.csv \
        --output combined_table.tex

Example:
    uv run python scripts/combine_summary_tables.py \
        outputs/benchmark_*_replogle22k562/*/additional_results/multimodel_summary_table_nondeg.csv \
        outputs/benchmark_*_wessels23/*/additional_results/multimodel_summary_table_nondeg.csv \
        --output combined_nondeg.tex
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Metric display names and whether higher is better
# (must match plot_multimodel_summary.py)
METRIC_INFO = {
    'mse': {'display': 'MSE', 'higher_better': False},
    'mse_degs': {'display': 'MSE(DEGs)', 'higher_better': False},
    'wmse': {'display': 'WMSE', 'higher_better': False},
    'mae': {'display': 'MAE', 'higher_better': False},
    'mae_degs': {'display': 'MAE(DEGs)', 'higher_better': False},
    'wmae': {'display': 'WMAE', 'higher_better': False},
    'pearson_deltactrl': {'display': 'Pearson(Δ Ctrl)', 'higher_better': True},
    'pearson_deltactrl_degs': {'display': 'Pearson(Δ Ctrl DEG)', 'higher_better': True},
    'pearson_deltapert': {'display': 'Pearson(Δ Pert)', 'higher_better': True},
    'pearson_deltapert_degs': {'display': 'Pearson(Δ Pert DEG)', 'higher_better': True},
    'r2_deltactrl': {'display': 'R²(Δ Ctrl)', 'higher_better': True},
    'r2_deltactrl_degs': {'display': 'R²(Δ Ctrl DEG)', 'higher_better': True},
    'r2_deltapert': {'display': 'R²(Δ Pert)', 'higher_better': True},
    'r2_deltapert_degs': {'display': 'R²(Δ Pert DEG)', 'higher_better': True},
    'weighted_r2_deltactrl': {'display': 'WR²(Δ Ctrl)', 'higher_better': True},
    'weighted_r2_deltapert': {'display': 'WR²(Δ Pert)', 'higher_better': True},
    'nir': {'display': 'NIR', 'higher_better': True},
    'pds': {'display': 'PDS', 'higher_better': True},
}

# Build reverse lookup: display name -> metric key
DISPLAY_TO_KEY = {v['display']: k for k, v in METRIC_INFO.items()}

BASELINE_KEYWORDS = [
    'control_mean', 'technical_duplicate', 'interpolated_duplicate',
    'additive', 'dataset_mean', 'linear', 'baselines', 'ground_truth',
]

# Positive controls: excluded from "best" bolding
POSITIVE_CONTROLS = ['technical_duplicate', 'interpolated_duplicate']


def format_model_name(model_name: str) -> str:
    """Transform model names to nice display format."""
    name_map = {
        'dataset_mean': 'Dataset Mean',
        'control_mean': 'Control Mean',
        'technical_duplicate': 'Tech. Dup.',
        'interpolated_duplicate': 'Inter. Dup.',
        'additive': 'Additive',
        'linear': 'Linear',
        'ground_truth': 'Ground Truth',
        'baselines': 'Baselines',
        'presage': 'PRESAGE',
        'sclambda': 'scLambda',
        'scgpt': 'scGPT',
        'fmlp_genept': 'fMLP-GenePT',
        'fmlp_esm2': 'fMLP-ESM2',
        'fmlp_geneformer': 'fMLP-Geneformer',
        'fmlp_scgpt': 'fMLP-scGPT',
        'gears': 'GEARS',
        'cellflow': 'CellFlow',
    }
    return name_map[model_name]


def extract_dataset_name(csv_path: Path) -> str:
    """Extract dataset name from CSV path.

    Expected path structure: .../benchmark_{models}_{dataset}/{timestamp}/additional_results/...
    """
    # Walk up to find the benchmark directory
    parts = csv_path.resolve().parts
    for part in parts:
        if part.startswith('benchmark_'):
            return part.split('_')[-1]
    # Fallback: use grandparent's grandparent
    return csv_path.parent.parent.parent.name.split('_')[-1]


def parse_mean_sem(cell_text: str):
    """Parse a 'mean ± sem' string into (mean_float, original_text).

    Returns (np.nan, cell_text) if the cell cannot be parsed (e.g. 'nan ± nan').
    """
    cell_text = cell_text.strip()
    match = re.match(r'^(-?\d+\.?\d*)\s*±\s*(-?\d+\.?\d*)$', cell_text)
    if match:
        return float(match.group(1)), cell_text
    return np.nan, cell_text


def _format_cell(cell_text: str, should_bold: bool) -> str:
    """Format a 'mean ± sem' cell for LaTeX display (truncated to 3 decimal places)."""
    _, raw_text = parse_mean_sem(cell_text)
    match = re.match(r'^(-?\d+\.?\d*)\s*±\s*(-?\d+\.?\d*)$', raw_text)
    if match:
        formatted = f"{float(match.group(1)):.3f} ± {float(match.group(2)):.3f}"
    elif 'nan' in raw_text.lower():
        formatted = "-"
    else:
        formatted = raw_text

    if should_bold:
        return f"\\textbf{{{formatted}}}"
    return formatted


def load_csv(csv_path: Path) -> tuple:
    """Load a summary CSV and return (dataset_name, df with raw model names, metric_display_names)."""
    df = pd.read_csv(csv_path)
    # Remove the deg_recovery_deltapert and pathway_recovery_deltapert columns
    if 'Pathway Recovery(Δ Pert)' in df.columns:
        df = df.drop(columns=['Pathway Recovery(Δ Pert)'])
    if 'DEG Recovery(Δ Pert)' in df.columns:
        df = df.drop(columns=['DEG Recovery(Δ Pert)'])
    # Rename Weighted MAE to WMAE
    df = df.rename(columns={'Weighted MAE': 'WMAE'})
    dataset_name = extract_dataset_name(csv_path)
    metric_columns = [c for c in df.columns if c != 'Model']
    return dataset_name, df, metric_columns


def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple multimodel summary CSV tables into a single LaTeX table.'
    )
    parser.add_argument(
        'csv_paths',
        nargs='+',
        type=str,
        help='Paths to summary CSV files (in display order)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for the combined LaTeX table (.tex)',
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv_paths]
    output_path = Path(args.output)

    # Load all datasets
    datasets = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        datasets.append(load_csv(csv_path))

    # Determine common metric columns (preserve order from first file)
    first_metrics = datasets[0][2]
    common_metrics = []
    for m in first_metrics:
        if all(m in ds[2] for ds in datasets):
            common_metrics.append(m)

    if not common_metrics:
        print("Error: No common metrics found across CSV files")
        sys.exit(1)

    print(f"Combining {len(datasets)} datasets with {len(common_metrics)} common metrics")
    for ds_name, df, _ in datasets:
        print(f"  - {ds_name}: {len(df)} models")

    # For each dataset, parse mean values and determine best model per metric
    # (excluding positive controls)
    dataset_records = []
    for dataset_name, df, _ in datasets:
        models_raw = df['Model'].tolist()

        # Parse mean values for bolding determination
        mean_values = {}  # (model, metric_display) -> float
        for _, row in df.iterrows():
            model = row['Model']
            for metric_display in common_metrics:
                mean_val, _ = parse_mean_sem(str(row[metric_display]))
                mean_values[(model, metric_display)] = mean_val

        # Determine best model per metric (excluding positive controls)
        # Uses full-precision floats parsed from the CSV (8 decimal places)
        best_model_per_metric = {}
        for metric_display in common_metrics:
            metric_key = DISPLAY_TO_KEY[metric_display]
            higher_better = METRIC_INFO[metric_key]['higher_better']

            eligible_values = []
            for model in models_raw:
                if model in POSITIVE_CONTROLS:
                    continue
                val = mean_values[(model, metric_display)]
                if not np.isnan(val):
                    eligible_values.append((model, val))

            if eligible_values:
                if higher_better:
                    best_model, _ = max(eligible_values, key=lambda x: x[1])
                else:
                    best_model, _ = min(eligible_values, key=lambda x: x[1])
                best_model_per_metric[metric_display] = best_model
            else:
                best_model_per_metric[metric_display] = None

        # Separate into trained models and baselines
        trained = [m for m in models_raw if m not in BASELINE_KEYWORDS]
        baselines = [m for m in models_raw if m in BASELINE_KEYWORDS]
        trained.sort()
        baselines.sort()

        dataset_records.append({
            'name': dataset_name,
            'df': df,
            'trained': trained,
            'baselines': baselines,
            'best_model_per_metric': best_model_per_metric,
        })

    # Build LaTeX table
    n_metrics = len(common_metrics)

    # Build header row with direction arrows
    header_parts = []
    for metric_display in common_metrics:
        metric_key = DISPLAY_TO_KEY[metric_display]
        arrow = '↑' if METRIC_INFO[metric_key]['higher_better'] else '↓'
        header_parts.append(f"\\textbf{{{metric_display} {arrow}}}")

    lines = []
    lines.append("\\begin{table*}[]")
    lines.append("\\centering")
    lines.append("\\caption{Combined model performance across datasets.}")
    lines.append("\\label{tab:combined-results}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    col_spec = "@{}lll" + "c" * n_metrics + "@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_line = (
        "\\textbf{Dataset} & \\textbf{Type} & \\textbf{Model} & "
        + " & ".join(header_parts)
        + " \\\\"
    )
    lines.append(header_line)
    lines.append("\\midrule")

    for ds_idx, ds in enumerate(dataset_records):
        trained = ds['trained']
        baselines = ds['baselines']
        best_model_per_metric = ds['best_model_per_metric']
        total_rows = len(trained) + len(baselines)

        dataset_display = ds['name']

        # Trained models
        for i, model in enumerate(trained):
            # Dataset cell: multirow on first row of this dataset
            if i == 0:
                dataset_cell = f"\\multirow{{{total_rows}}}{{*}}{{{dataset_display}}}"
            else:
                dataset_cell = ""

            # Type cell: multirow on first row of trained section
            if i == 0:
                type_cell = f"\\multirow{{{len(trained)}}}{{*}}{{Model}}"
            else:
                type_cell = ""

            model_display = format_model_name(model)

            # Build metric cells
            metric_cells = []
            for metric_display in common_metrics:
                cell_text = str(ds['df'].loc[ds['df']['Model'] == model, metric_display].iloc[0])

                # Bold if this model is the best for this metric
                should_bold = (
                    model not in POSITIVE_CONTROLS
                    and best_model_per_metric[metric_display] == model
                )

                metric_cells.append(_format_cell(cell_text, should_bold))

            row = f"{dataset_cell} & {type_cell} & {model_display} & " + " & ".join(metric_cells) + " \\\\"
            lines.append(row)

        # Separator between trained and baselines
        lines.append(f"\\cmidrule(l){{2-{3 + n_metrics}}}")

        # Baselines
        for i, model in enumerate(baselines):
            # Dataset cell: empty (already covered by multirow)
            dataset_cell = ""

            # Type cell: multirow on first row of baseline section
            if i == 0:
                type_cell = f"\\multirow{{{len(baselines)}}}{{*}}{{Baseline}}"
            else:
                type_cell = ""

            model_display = format_model_name(model)

            # Build metric cells
            metric_cells = []
            for metric_display in common_metrics:
                cell_text = str(ds['df'].loc[ds['df']['Model'] == model, metric_display].iloc[0])

                # Bold if this model is the best for this metric
                should_bold = (
                    model not in POSITIVE_CONTROLS
                    and best_model_per_metric[metric_display] == model
                )

                metric_cells.append(_format_cell(cell_text, should_bold))

            row = f"{dataset_cell} & {type_cell} & {model_display} & " + " & ".join(metric_cells) + " \\\\"
            lines.append(row)

        # Separator between datasets (midrule), but not after last dataset
        if ds_idx < len(dataset_records) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Saved combined LaTeX table to: {output_path}")
    print(f"  {len(datasets)} datasets × {len(common_metrics)} metrics")


if __name__ == '__main__':
    main()
