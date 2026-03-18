# %% Imports

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# %% Resolve project root for both script and interactive execution
try:
    _script_dir = Path(__file__).resolve().parent
except NameError:  # __file__ not defined in interactive/notebook environments
    _script_dir = Path.cwd()
# Navigate to project root (handles both root and perturbation_strength as cwd)
PROJECT_ROOT = _script_dir if (_script_dir / "data").exists() else _script_dir.parents[1]
FIGURES_DIR = PROJECT_ROOT / "analyses/perturbation_strength/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% Load norman19 data

data_path = PROJECT_ROOT / "data/norman19/norman19_processed_complete.h5ad"
print(f"Loading data from: {data_path}")
adata = sc.read_h5ad(data_path)
print(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# %% Compute DEG counts per perturbation and categorize into terciles

deg_gene_dict = adata.uns['deg_gene_dict']

# Get DEG counts for all perturbations (excluding controls)
deg_counts = pd.Series({
    pert_name: len(deg_genes)
    for pert_name, deg_genes in deg_gene_dict.items()
    if 'control' not in pert_name.lower()
})
deg_counts = deg_counts.sort_values(ascending=True)

print(f"Total perturbations: {len(deg_counts)}")
print(f"DEG count range: {deg_counts.min()} - {deg_counts.max()}")

# Split into terciles
n = len(deg_counts)
tercile_size = n // 3

weak_perts = deg_counts.iloc[:tercile_size]
medium_perts = deg_counts.iloc[tercile_size:2 * tercile_size]
strong_perts = deg_counts.iloc[2 * tercile_size:]

print(f"\nWeak tercile ({len(weak_perts)} perts): {weak_perts.min()}-{weak_perts.max()} DEGs")
print(f"Medium tercile ({len(medium_perts)} perts): {medium_perts.min()}-{medium_perts.max()} DEGs")
print(f"Strong tercile ({len(strong_perts)} perts): {strong_perts.min()}-{strong_perts.max()} DEGs")

# %% Randomly select 2 perturbations from each tercile

np.random.seed(42)

selected_weak = np.random.choice(weak_perts.index, size=2, replace=False)
selected_medium = np.random.choice(medium_perts.index, size=2, replace=False)
selected_strong = np.random.choice(strong_perts.index, size=2, replace=False)

print("\nSelected perturbations:")
for label, selected in [("Weak", selected_weak), ("Medium", selected_medium), ("Strong", selected_strong)]:
    for pert in selected:
        print(f"  {label}: {pert} ({deg_counts[pert]} DEGs)")

# %% Plot 2x3 grid of alpha distributions

# Columns = weak, medium, strong; Rows = 2 perturbations per category
fig, axes = plt.subplots(2, 3, figsize=(20, 8))

categories = [
    ("Weak", selected_weak),
    ("Medium", selected_medium),
    ("Strong", selected_strong),
]

pvals_dict = adata.uns['pvals_adj_df_dict']
names_dict = adata.uns['names_df_dict']

for col_idx, (category_label, selected) in enumerate(categories):
    for row_idx, perturbation in enumerate(selected):
        ax = axes[row_idx, col_idx]

        # Get adjusted p-values and gene names
        pvals = pvals_dict[perturbation]
        gene_names = names_dict[perturbation]

        # Compute alpha = 1 - padj
        pvals_series = pd.Series(pvals, index=gene_names)
        pvals_series_sorted = pvals_series.sort_values(ascending=False)
        rank = np.arange(len(pvals_series_sorted))
        alpha_values = 1 - pvals_series_sorted.values

        # Plot ranking curve with fill
        ax.plot(rank, alpha_values, color='darkblue', linewidth=2, alpha=0.9)
        ax.fill_between(rank, 0, alpha_values, color='blue', alpha=0.3)

        # Reference lines at DEG thresholds
        ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

        # Axis limits
        ax.set_xlim([0, len(pvals_series_sorted)])
        ax.set_ylim([0, 1.05])

        # Title with perturbation name and DEG count
        pert_display = perturbation.replace("norman19_", "")
        n_degs = deg_counts[perturbation]
        ax.set_title(f'{pert_display}\n{n_degs} DEGs', fontsize=18, fontweight='bold')

        # Axis labels
        ax.set_xlabel('Gene Rank', fontsize=18)
        if col_idx == 0:
            ax.set_ylabel('Alpha (1 - adjusted p-value)', fontsize=18)

        # Styling
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "alpha_distributions_norman19.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"\nFigure saved to: {FIGURES_DIR / 'alpha_distributions_norman19.png'}")

# %%
