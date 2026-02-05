"""
Batch UMAP Visualization

Loads perturbation datasets, computes PCA and UMAP, and visualizes
cells colored by batch to assess batch effects. Also performs clustering
and calculates cluster-batch association to quantify manifold confounding.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Set the working directory to the root of the project
import os
os.chdir(Path(__file__).parent.parent)

# Define datasets to analyze with their batch column names
DATASETS = {
    'wessels23': 'HTO',
    'norman19': 'gemgroup',
    'replogle22k562': 'batch',
    'nadig25hepg2': 'gem_group',
}

# Output directory for figures
OUTPUT_DIR = Path("analyses/batch_umap_figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def calculate_cluster_batch_association(adata: sc.AnnData, batch_col: str) -> dict:
    """
    Calculate association metrics between clusters and batch labels.
    
    Returns metrics quantifying how much the data manifold is confounded by batch.
    """
    clusters = adata.obs['leiden'].astype(str)
    batches = adata.obs[batch_col].astype(str)
    
    # Adjusted Rand Index: measures similarity between two clusterings
    # ARI = 0 means random, ARI = 1 means perfect agreement
    ari = adjusted_rand_score(clusters, batches)
    
    # Normalized Mutual Information: how much knowing one tells you about the other
    # NMI = 0 means independent, NMI = 1 means perfectly predictive
    nmi = normalized_mutual_info_score(clusters, batches)
    
    # Additional metrics
    n_clusters = adata.obs['leiden'].nunique()
    n_batches = adata.obs[batch_col].nunique()
    
    # Batch purity per cluster: for each cluster, what fraction of cells come from the dominant batch?
    contingency = pd.crosstab(clusters, batches)
    batch_purity_per_cluster = contingency.max(axis=1) / contingency.sum(axis=1)
    mean_batch_purity = batch_purity_per_cluster.mean()
    
    return {
        'n_clusters': n_clusters,
        'n_batches': n_batches,
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'mean_batch_purity_per_cluster': mean_batch_purity,
    }


MAX_CELLS = 20000  # Maximum cells to use for faster processing


def load_and_process_dataset(dataset_name: str, batch_col: str) -> sc.AnnData:
    """Load dataset and compute PCA/UMAP."""
    data_path = f"data/{dataset_name}/{dataset_name}_processed_complete.h5ad"
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print(f"  Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    print(f"  Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Filter out cells with NaN in batch column
    n_before = adata.n_obs
    valid_mask = adata.obs[batch_col].notna()
    adata = adata[valid_mask].copy()
    n_after = adata.n_obs
    if n_before != n_after:
        print(f"  Excluded {n_before - n_after} cells with NaN in '{batch_col}'")
    
    # Exclude control cells
    n_before = adata.n_obs
    ctrl_mask = adata.obs['condition'].str.contains('ctrl|control', case=False, na=False)
    adata = adata[~ctrl_mask].copy()
    n_after = adata.n_obs
    if n_before != n_after:
        print(f"  Excluded {n_before - n_after} control cells")
    
    # Downsample if too many cells
    if adata.n_obs > MAX_CELLS:
        print(f"  Downsampling from {adata.n_obs} to {MAX_CELLS} cells...")
        sc.pp.subsample(adata, n_obs=MAX_CELLS, random_state=42)
    
    # Ensure batch column is categorical (not continuous)
    adata.obs[batch_col] = adata.obs[batch_col].astype(str).astype('category')

    
    
    print(f"  Final: {adata.n_obs} cells")
    
    # Check if PCA already exists
    if 'X_pca' not in adata.obsm:
        print("  Computing PCA...")
        sc.pp.pca(adata, n_comps=50)
    else:
        print("  PCA already computed, using existing")
    
    # Check if neighbors already computed
    if 'neighbors' not in adata.uns:
        print("  Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    else:
        print("  Neighbors already computed, using existing")
    
    # Check if UMAP already exists
    if 'X_umap' not in adata.obsm:
        print("  Computing UMAP...")
        sc.tl.umap(adata, random_state=42)
    else:
        print("  UMAP already computed, using existing")
    
    # Compute Leiden clustering
    if 'leiden' not in adata.obs.columns:
        print("  Computing Leiden clustering...")
        sc.tl.leiden(adata, resolution=1.0)
    else:
        print("  Leiden clustering already computed, using existing")
    
    n_clusters = adata.obs['leiden'].nunique()
    print(f"  Found {n_clusters} clusters")
    
    return adata


def plot_umap_by_batch(adata: sc.AnnData, dataset_name: str, batch_col: str):
    """Plot UMAP colored by batch."""
    print(f"  Plotting UMAP colored by '{batch_col}'...")
    
    # Get number of unique batches for color palette
    n_batches = adata.obs[batch_col].nunique()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sc.pl.umap(
        adata,
        color=batch_col,
        ax=ax,
        show=False,
        title=f"{dataset_name}: UMAP colored by {batch_col} ({n_batches} batches)",
        legend_loc=None,
        size=20,
        palette='Dark2',
    )
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{dataset_name}_umap_by_batch.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def plot_umap_batch_and_clusters(adata: sc.AnnData, dataset_name: str, batch_col: str, metrics: dict):
    """Plot UMAP colored by batch and clusters side by side with metrics."""
    print(f"  Plotting UMAP with batch and clusters comparison...")
    
    n_batches = adata.obs[batch_col].nunique()
    n_clusters = adata.obs['leiden'].nunique()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot by batch
    sc.pl.umap(
        adata,
        color=batch_col,
        ax=axes[0],
        show=False,
        title=f"Colored by {batch_col} ({n_batches} batches)",
        legend_loc=None,
        size=15,
        palette='Dark2',
    )
    
    # Plot by clusters
    sc.pl.umap(
        adata,
        color='leiden',
        ax=axes[1],
        show=False,
        title=f"Colored by Leiden clusters ({n_clusters} clusters)",
        legend_loc='right margin' if n_clusters <= 20 else None,
        legend_fontsize=8,
        size=15,
    )
    
    # Add metrics as text
    metrics_text = (
        f"Cluster-Batch Association Metrics:\n"
        f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}\n"
        f"  Normalized MI: {metrics['normalized_mutual_info']:.3f}\n"
        f"  Mean batch purity/cluster: {metrics['mean_batch_purity_per_cluster']:.3f}"
    )
    
    fig.text(0.5, -0.02, metrics_text, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    plt.suptitle(f"{dataset_name}: Batch vs Cluster Structure ({adata.n_obs} cells)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{dataset_name}_batch_cluster_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def plot_umap_by_condition(adata: sc.AnnData, dataset_name: str):
    """Plot UMAP colored by condition (for reference)."""
    print(f"  Plotting UMAP colored by 'condition'...")
    
    n_conditions = adata.obs['condition'].nunique()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For many conditions, don't show legend
    show_legend = n_conditions <= 30
    
    sc.pl.umap(
        adata,
        color='condition',
        ax=ax,
        show=False,
        title=f"{dataset_name}: UMAP colored by condition ({n_conditions} conditions)",
        legend_loc='right margin' if show_legend else None,
        legend_fontsize=6,
        size=15,
    )
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{dataset_name}_umap_by_condition.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def main():
    print("="*70)
    print("BATCH UMAP VISUALIZATION & CLUSTER-BATCH ANALYSIS")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_metrics = []
    
    # Process each dataset individually
    for dataset_name, batch_col in DATASETS.items():
        adata = load_and_process_dataset(dataset_name, batch_col)
        
        # Calculate cluster-batch association metrics
        metrics = calculate_cluster_batch_association(adata, batch_col)
        metrics['dataset'] = dataset_name
        metrics['batch_column'] = batch_col
        all_metrics.append(metrics)
        
        # Print metrics
        print(f"\n  Cluster-Batch Association Metrics:")
        print(f"    Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
        print(f"    Normalized MI: {metrics['normalized_mutual_info']:.4f}")
        print(f"    Mean batch purity per cluster: {metrics['mean_batch_purity_per_cluster']:.4f}")
        
        # Generate plots
        plot_umap_by_batch(adata, dataset_name, batch_col)
        plot_umap_batch_and_clusters(adata, dataset_name, batch_col, metrics)
        plot_umap_by_condition(adata, dataset_name)
        
        # Clean up memory
        del adata
    
    # Summary table
    print("\n\n" + "="*70)
    print("CLUSTER-BATCH CONFOUNDING SUMMARY")
    print("="*70)
    
    metrics_df = pd.DataFrame(all_metrics)
    col_order = [
        'dataset', 'n_clusters', 'n_batches',
        'adjusted_rand_index', 'normalized_mutual_info',
        'mean_batch_purity_per_cluster'
    ]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]
    
    print("\n" + metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "cluster_batch_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Adjusted Rand Index (ARI):
  - 0: Cluster and batch assignments are random/independent
  - 1: Cluster and batch assignments are identical
  - High ARI = clusters align with batches = batch confounding in manifold

Normalized Mutual Information (NMI):
  - 0: Knowing cluster tells nothing about batch
  - 1: Cluster perfectly predicts batch
  - High NMI = batch structure dominates the data manifold

Mean Batch Purity per Cluster:
  - High value (close to 1) = each cluster is dominated by one batch (bad!)
  - Low value = clusters contain cells from many batches (good!)
""")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"All figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
