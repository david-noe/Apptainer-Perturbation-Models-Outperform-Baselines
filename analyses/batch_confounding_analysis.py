"""
Batch Confounding Analysis

Analyzes the degree to which batch effects are confounded with perturbation labels
in perturbation response datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import h5py
from anndata.experimental import read_elem

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


def load_obs_only(data_path: str) -> pd.DataFrame:
    """Load only the observation metadata without the full expression matrix."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    with h5py.File(path, 'r') as f:
        obs = read_elem(f['obs'])
    
    return obs


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Calculate Cramer's V statistic for categorical-categorical association.
    
    Cramer's V ranges from 0 (no association) to 1 (perfect association).
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    
    if min_dim == 0:
        return 0.0
    
    return np.sqrt(chi2 / (n * min_dim))


def normalized_mutual_info(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate normalized mutual information between two categorical variables.
    
    NMI ranges from 0 (independent) to 1 (perfectly predictive).
    """
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(x, y)


def analyze_batch_condition_confounding(obs: pd.DataFrame, dataset_name: str, batch_col: str) -> dict:
    """
    Analyze the confounding between batch and condition in a dataset.
    
    Returns a dictionary with various confounding metrics.
    """
    results = {'dataset': dataset_name, 'batch_column': batch_col}
    
    # Basic counts
    n_conditions = obs['condition'].nunique()
    n_batches = obs[batch_col].nunique()
    n_cells = len(obs)
    
    results['n_conditions'] = n_conditions
    results['n_batches'] = n_batches
    results['n_cells'] = n_cells
    
    # Create contingency table
    contingency = pd.crosstab(obs['condition'], obs[batch_col])
    breakpoint()
    
    # Cramer's V - measures association strength
    results['cramers_v'] = cramers_v(contingency)
    
    # Normalized Mutual Information
    results['normalized_mutual_info'] = normalized_mutual_info(obs['condition'], obs[batch_col])
    
    # Coverage metrics
    # How many batches does each condition appear in?
    batches_per_condition = (contingency > 0).sum(axis=1)
    results['mean_batches_per_condition'] = batches_per_condition.median()
    results['min_batches_per_condition'] = batches_per_condition.min()
    results['max_batches_per_condition'] = batches_per_condition.max()
    results['pct_conditions_single_batch'] = (batches_per_condition == 1).median() * 100
    
    # How many conditions does each batch contain?
    conditions_per_batch = (contingency > 0).sum(axis=0)
    results['mean_conditions_per_batch'] = conditions_per_batch.median()
    results['min_conditions_per_batch'] = conditions_per_batch.min()
    results['max_conditions_per_batch'] = conditions_per_batch.max()
    
    # Perfect confounding would be: each condition appears in exactly one batch
    # (i.e., condition perfectly predicts batch)
    # This is measured by pct_conditions_single_batch
    
    # Sparsity of the contingency table
    # How many condition-batch pairs actually have cells?
    n_nonempty_pairs = (contingency > 0).sum().sum()
    n_possible_pairs = n_conditions * n_batches
    results['contingency_density'] = n_nonempty_pairs / n_possible_pairs
    
    return results, contingency, batches_per_condition


def print_detailed_analysis(obs: pd.DataFrame, dataset_name: str, batch_col: str,
                           contingency: pd.DataFrame, batches_per_condition: pd.Series):
    """Print detailed analysis of batch-condition relationships."""
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {dataset_name} (batch column: '{batch_col}')")
    print(f"{'='*60}")
    
    # Conditions that only appear in one batch (most confounded)
    single_batch_conditions = batches_per_condition[batches_per_condition == 1]
    if len(single_batch_conditions) > 0:
        print(f"\nConditions appearing in only ONE batch ({len(single_batch_conditions)} total):")
        # Show a sample
        sample_size = min(10, len(single_batch_conditions))
        for cond in single_batch_conditions.head(sample_size).index:
            batch = contingency.loc[cond][contingency.loc[cond] > 0].index[0]
            n_cells_cond = contingency.loc[cond, batch]
            print(f"  {cond}: batch={batch}, n_cells={n_cells_cond}")
        if len(single_batch_conditions) > sample_size:
            print(f"  ... and {len(single_batch_conditions) - sample_size} more")
    
    # Conditions that appear in many batches (least confounded)
    multi_batch_conditions = batches_per_condition[batches_per_condition > 1].sort_values(ascending=False)
    if len(multi_batch_conditions) > 0:
        print(f"\nConditions appearing in MULTIPLE batches ({len(multi_batch_conditions)} total):")
        sample_size = min(10, len(multi_batch_conditions))
        for cond in multi_batch_conditions.head(sample_size).index:
            n_batches_cond = multi_batch_conditions[cond]
            batches_list = contingency.loc[cond][contingency.loc[cond] > 0].index.tolist()
            print(f"  {cond}: {n_batches_cond} batches -> {batches_list[:5]}{'...' if len(batches_list) > 5 else ''}")
        if len(multi_batch_conditions) > sample_size:
            print(f"  ... and {len(multi_batch_conditions) - sample_size} more")
    
    # Batch size distribution
    print(f"\nBatch sizes ({batch_col}):")
    batch_sizes = obs[batch_col].value_counts().sort_index()
    print(f"  Min: {batch_sizes.min()}, Max: {batch_sizes.max()}, Mean: {batch_sizes.mean():.1f}")
    
    # Condition size distribution (excluding controls)
    non_ctrl_conditions = obs[~obs['condition'].str.contains('ctrl', case=False, na=False)]
    if len(non_ctrl_conditions) > 0:
        cond_sizes = non_ctrl_conditions['condition'].value_counts()
        print(f"\nCells per condition (excluding controls):")
        print(f"  Min: {cond_sizes.min()}, Max: {cond_sizes.max()}, Mean: {cond_sizes.mean():.1f}")


def main():
    print("="*70)
    print("BATCH-PERTURBATION CONFOUNDING ANALYSIS")
    print("="*70)
    
    all_results = []
    
    for dataset_name, batch_col in DATASETS.items():
        print(f"\n\nProcessing {dataset_name} (batch column: '{batch_col}')...")
        
        # Construct data path
        data_path = f"data/{dataset_name}/{dataset_name}_processed_complete.h5ad"
        
        # Load obs
        obs = load_obs_only(data_path)
        print(f"  Loaded {len(obs)} cells")
        
        # Check if batch column exists
        if batch_col not in obs.columns:
            print(f"  WARNING: '{batch_col}' column not found in {dataset_name}")
            print(f"  Available columns: {list(obs.columns)}")
            continue
        
        if 'condition' not in obs.columns:
            print(f"  WARNING: 'condition' column not found in {dataset_name}")
            continue
        
        # Filter out rows with NaN in condition or batch columns
        n_before = len(obs)
        obs = obs.dropna(subset=['condition', batch_col])
        n_after = len(obs)
        if n_before != n_after:
            print(f"  Excluded {n_before - n_after} cells with NaN values in 'condition' or '{batch_col}'")
        
        # Exclude control cells
        n_before = len(obs)
        obs = obs[~obs['condition'].str.contains('ctrl|control', case=False, na=False)]
        n_after = len(obs)
        if n_before != n_after:
            print(f"  Excluded {n_before - n_after} control cells")
        
        # Analyze confounding
        results, contingency, batches_per_condition = analyze_batch_condition_confounding(obs, dataset_name, batch_col)
        all_results.append(results)
        
        # Print summary
        print(f"\n  Summary for {dataset_name}:")
        print(f"    Batch column: {batch_col}")
        print(f"    N conditions: {results['n_conditions']}")
        print(f"    N batches: {results['n_batches']}")
        print(f"    N cells: {results['n_cells']}")
        print(f"    Cramer's V: {results['cramers_v']:.4f}")
        print(f"    Normalized MI: {results['normalized_mutual_info']:.4f}")
        print(f"    Mean batches per condition: {results['mean_batches_per_condition']:.2f}")
        print(f"    % conditions in single batch: {results['pct_conditions_single_batch']:.1f}%")
        print(f"    Contingency table density: {results['contingency_density']:.4f}")
        
        # Print detailed analysis
        print_detailed_analysis(obs, dataset_name, batch_col, contingency, batches_per_condition)
    
    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    col_order = [
        'dataset', 'batch_column', 'n_conditions', 'n_batches', 'n_cells',
        'cramers_v', 'normalized_mutual_info',
        'mean_batches_per_condition', 'pct_conditions_single_batch',
        'contingency_density'
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]
    
    print("\n" + results_df.to_string(index=False))
    
    # Interpretation
    print("\n\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Cramer's V:
  - 0.0: No association (batch and condition are independent)
  - 1.0: Perfect association (batch perfectly predicts condition or vice versa)
  - < 0.1: Negligible confounding
  - 0.1-0.3: Weak confounding  
  - 0.3-0.5: Moderate confounding
  - > 0.5: Strong confounding

% Conditions in Single Batch:
  - Higher = more confounded (each perturbation only seen in one batch)
  - Lower = less confounded (perturbations spread across batches)

Contingency Density:
  - Low density with high Cramer's V = highly structured/confounded design
  - High density = more balanced experimental design
""")
    
    # Save results
    output_path = Path("analyses/batch_confounding_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
