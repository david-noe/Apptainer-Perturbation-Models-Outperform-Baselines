"""
Weight Distribution Analysis for WMSE/WMAE/WR² Metrics

For each dataset, selects perturbations with high/medium/low DEG counts
and plots the histogram of gene weights to understand how many genes
receive meaningful weight in the weighted metrics.

The weighting scheme:
1. Takes absolute value of DEG test statistics (scores)
2. Min-max normalizes across genes
3. Squares the normalized values (for stronger emphasis on DEGs)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import sys

# Set the working directory to the root of the project
import os
os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent))

from cellsimbench.core.data_manager import DataManager

# Define datasets to analyze
DATASETS = ['norman19', 'wessels23', 'replogle22k562', 'nadig25hepg2']
DATASETS = ['replogle22k562']

# Output directory for figures
OUTPUT_DIR = Path("analyses/weight_distribution_figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def count_significant_degs(pvals: np.ndarray, pval_threshold: float = 0.05) -> int:
    """Count the number of significant DEGs for a perturbation."""
    return np.sum(pvals < pval_threshold)


def select_perturbations(
    deg_pvals_dict: Dict[str, np.ndarray],
    n_select: int = 9,
    random_seed: int = 2,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Randomly select perturbations.
    
    Args:
        deg_pvals_dict: Dictionary mapping perturbation keys to p-value arrays.
        n_select: Number of perturbations to select.
        random_seed: Random seed for reproducibility.
    
    Returns list of selected perturbation names and dict of DEG counts.
    """
    # Count DEGs for each perturbation
    deg_counts = {}
    for pert_key, pvals in deg_pvals_dict.items():
        if 'control' in pert_key.lower() or 'ctrl' in pert_key.lower():
            continue
        deg_counts[pert_key] = count_significant_degs(pvals)
    
    all_perts = list(deg_counts.keys())
    
    if len(all_perts) < n_select:
        raise ValueError(f"Need at least {n_select} perturbations, found {len(all_perts)}")
    
    # Randomly select perturbations
    rng = np.random.default_rng(random_seed)
    selected = list(rng.choice(all_perts, size=n_select, replace=False))
    
    return selected, deg_counts


def analyze_weight_distribution(weights: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics about weight distribution."""
    total_genes = len(weights)
    
    # Count genes above various thresholds
    stats = {
        'total_genes': total_genes,
        'nonzero': np.sum(weights > 0),
        'gt_0.001': np.sum(weights > 0.001),
        'gt_0.01': np.sum(weights > 0.01),
        'gt_0.05': np.sum(weights > 0.05),
        'gt_0.1': np.sum(weights > 0.1),
        'gt_0.25': np.sum(weights > 0.25),
        'gt_0.5': np.sum(weights > 0.5),
        'max_weight': np.max(weights),
        'mean_weight': np.mean(weights),
        'median_weight': np.median(weights),
        # Effective number of genes (inverse of Herfindahl index)
        'effective_n_genes': 1.0 / np.sum((weights / weights.sum())**2) if weights.sum() > 0 else 0,
    }
    
    return stats


def plot_weight_rank(
    weights: np.ndarray,
    dataset_name: str,
    pert_name: str,
    n_degs: int,
    ax: plt.Axes
) -> Dict[str, float]:
    """Plot rank plot of weights (sorted highest to lowest) and return statistics."""
    # Normalize weights to sum to 1 (same as in WMSE/WMAE computation)
    normalized_weights = weights / weights.sum()
    
    stats = analyze_weight_distribution(normalized_weights)
    
    # Sort weights from highest to lowest
    sorted_weights = np.sort(normalized_weights)[::-1]
    ranks = np.arange(1, len(sorted_weights) + 1)
    
    ax.plot(ranks, sorted_weights, color='steelblue', linewidth=1.5)
    ax.fill_between(ranks, sorted_weights, alpha=0.3, color='steelblue')
    
    # Title with dataset, perturbation, and DEG count
    title = f"{pert_name}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Gene rank')
    ax.set_ylabel('Weight')
    ax.set_xscale('log', base=2)
    
    return stats


def process_dataset(dataset_name: str) -> Tuple[Dict, List[str]]:
    """Load dataset via DataManager and get weight distributions for selected perturbations."""
    data_path = f"data/{dataset_name}/{dataset_name}_processed_complete.h5ad"
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    
    # Create DataManager config and load dataset
    config = {'data_path': data_path, 'covariate_key': 'donor_id'}
    dm = DataManager(config)
    dm.load_dataset()
    
    print(f"  Loaded {dm.adata.n_obs} cells x {dm.adata.n_vars} genes")
    
    # Get DEG p-values for selecting perturbations
    deg_pvals_dict = dm.deg_pvals_dict
    
    print(f"  Found {len(dm.pert_normalized_abs_scores_vsrest)} perturbations with precomputed weights")
    
    # Select perturbations randomly
    selected, deg_counts = select_perturbations(deg_pvals_dict)
    
    print(f"\n  Selected perturbations:")
    for pert in selected:
        print(f"    {pert} ({deg_counts[pert]} DEGs)")
    
    # Get precomputed weights from DataManager for each selected perturbation
    gene_order = list(dm.adata.var_names)
    results = {}
    for pert in selected:
        # Parse covariate and condition from pert key
        parts = pert.split('_')
        covariate = parts[0]
        condition = '_'.join(parts[1:])
        
        # Get weights directly from DataManager (already precomputed)
        weights = dm.get_deg_weights(covariate, condition, gene_order)
        
        results[pert] = {
            'weights': weights,
            'n_degs': deg_counts[pert],
        }
    
    return results, gene_order


def main():
    print("="*70)
    print("WEIGHT DISTRIBUTION ANALYSIS FOR WMSE/WMAE/WR² METRICS")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Collect all results for summary
    all_stats = []
    
    for dataset_name in DATASETS:
        results, var_names = process_dataset(dataset_name)
        
        # Create figure for this dataset (3 rows x 3 perturbations)
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (pert, data) in enumerate(results.items()):
            ax = axes[idx]
            
            stats = plot_weight_rank(
                weights=data['weights'],
                dataset_name=dataset_name,
                pert_name=pert,
                n_degs=data['n_degs'],
                ax=ax
            )
            
            # Record stats
            stats['dataset'] = dataset_name
            stats['perturbation'] = pert
            stats['n_degs'] = data['n_degs']
            all_stats.append(stats)
        
        # Adjust layout and save this dataset's figure
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f"{dataset_name}_weight_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved figure to: {output_path}")
    
    # Create summary statistics table
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df[[
        'dataset', 'perturbation', 'n_degs',
        'total_genes', 'gt_0.01', 'gt_0.1', 'effective_n_genes'
    ]]
    stats_df.columns = [
        'Dataset', 'Perturbation', 'N DEGs (p<0.05)',
        'Total Genes', 'Genes w>0.01', 'Genes w>0.1', 'Effective N Genes'
    ]
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(stats_df.to_string(index=False))
    
    # Save statistics
    stats_path = OUTPUT_DIR / "weight_distribution_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved to: {stats_path}")
    
    # Print interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
The 'Effective N Genes' metric (inverse Herfindahl index) indicates how many 
genes receive substantial weight. A value of 100 means the weights are 
distributed as if 100 genes were weighted equally.

Key observations:
- Weights are derived from DEG test statistics (squared min-max normalized)
- Higher DEG counts generally lead to more genes with meaningful weights
- The squared normalization emphasizes the strongest DEGs
""")


if __name__ == "__main__":
    main()
