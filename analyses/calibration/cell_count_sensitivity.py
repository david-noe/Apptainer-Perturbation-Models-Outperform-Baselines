#!/usr/bin/env python
"""
Cell Count Sensitivity Analysis for Norman19.

Investigates how cell count per perturbation affects the technical duplicate
positive control and calibration (DRF). Starts from raw Norman19 data, filters
to perturbations with >= 400 cells, then for each target cell count
(10, 24, 50, 100, 200, 400) downsamples evenly from pre-assigned halves,
re-preprocesses from scratch (including DEGs), recomputes all baselines,
evaluates metrics, and shows how calibration degrades with fewer cells.

Three-stage cached pipeline:
  Stage 1: Preprocessing + baselines -> cached h5ad per cell count
  Stage 2: Metrics + DRF -> cached CSV
  Stage 3: Plotting -> PNG files

Usage:
    python analyses/calibration/cell_count_sensitivity.py
    python analyses/calibration/cell_count_sensitivity.py --force
    python analyses/calibration/cell_count_sensitivity.py --plot-only
    python analyses/calibration/cell_count_sensitivity.py --workers 4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, issparse
from scipy.stats import pearsonr
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cellsimbench.core.data_manager import mse, wmse, r2_score_on_deltas

# ============================================================================
# Configuration
# ============================================================================

RAW_DATA_PATH = PROJECT_ROOT / "data" / "norman19" / "norman19_downloaded.h5ad"
CACHE_DIR = PROJECT_ROOT / "analyses" / "calibration" / "cell_count_cache"
RESULTS_DIR = PROJECT_ROOT / "analyses" / "calibration" / "results"

CELL_COUNTS = [8, 16, 32, 64, 128, 256]
MIN_CELLS_PER_PERT = np.max(CELL_COUNTS)
MAX_CELLS_CONTROL = 8192
DATASET_NAME = "norman19"
N_TOP_GENES = 8192

METRICS_CSV = RESULTS_DIR / "cell_count_sensitivity.csv"
SUMMARY_CSV = RESULTS_DIR / "cell_count_sensitivity_summary.csv"


# ============================================================================
# Stage 1: Preprocessing + Baselines
# ============================================================================

def load_and_prepare_raw_data():
    """Load raw Norman19 data, filter, cap control, assign tech_dup_split,
    and remove perturbations with < 400 cells.

    Returns:
        adata: Prepared AnnData (raw counts, filtered, with tech_dup_split)
    """
    print("=" * 60)
    print("LOADING AND PREPARING RAW DATA")
    print("=" * 60)

    # 1. Load
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    adata = sc.read_h5ad(str(RAW_DATA_PATH))
    print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # 2. Standardize columns
    print("Standardizing columns...")
    adata.obs.rename(columns={
        'nCount_RNA': 'ncounts',
        'nFeature_RNA': 'ngenes',
        'percent.mt': 'percent_mito',
        'cell_line': 'cell_type',
    }, inplace=True)

    # 3. Standardize perturbation names
    adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('_', '+')
    adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
    adata.obs['condition'] = adata.obs.perturbation.copy()

    # 4. Add donor_id
    adata.obs['donor_id'] = DATASET_NAME

    # 5. Convert to sparse
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)

    # 6. Filter cells and genes on full raw data
    print("Filtering cells (min_genes=200) and genes (min_cells=3)...")
    n_before = adata.shape
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  Before: {n_before[0]} cells x {n_before[1]} genes")
    print(f"  After:  {adata.shape[0]} cells x {adata.shape[1]} genes")

    # 7. Cap control cells at 8192
    np.random.seed(42)
    ctrl_mask = adata.obs['condition'] == 'control'
    n_ctrl = ctrl_mask.sum()
    print(f"Control cells: {n_ctrl}")
    if n_ctrl > MAX_CELLS_CONTROL:
        ctrl_indices = adata.obs[ctrl_mask].index.tolist()
        keep_ctrl = np.random.choice(ctrl_indices, size=MAX_CELLS_CONTROL, replace=False)
        remove_ctrl = set(ctrl_indices) - set(keep_ctrl)
        adata = adata[~adata.obs.index.isin(remove_ctrl)].copy()
        print(f"  Capped control to {MAX_CELLS_CONTROL} cells")
    else:
        print(f"  Control cells <= {MAX_CELLS_CONTROL}, no capping needed")

    # 8. Assign one-time tech_dup_split (non-control only)
    print("Assigning one-time tech_dup_split...")
    adata.obs['tech_dup_split'] = pd.NA

    unique_conditions = adata.obs['condition'].unique()
    non_ctrl_conditions = [c for c in unique_conditions
                           if 'control' not in c.lower() and 'ctrl' not in c.lower()]

    n_split = 0
    for condition in non_ctrl_conditions:
        condition_cells = adata.obs[adata.obs['condition'] == condition].index
        if len(condition_cells) >= 2:
            cell_indices = np.random.permutation(condition_cells)
            split_idx = len(cell_indices) // 2
            adata.obs.loc[cell_indices[:split_idx], 'tech_dup_split'] = 'first_half'
            adata.obs.loc[cell_indices[split_idx:], 'tech_dup_split'] = 'second_half'
            n_split += 1

    print(f"  Assigned tech_dup_split for {n_split} perturbations")
    split_counts = adata.obs['tech_dup_split'].value_counts()
    print(f"  first_half: {split_counts.get('first_half', 0)}, "
          f"second_half: {split_counts.get('second_half', 0)}")

    # 9. Remove perturbations with < 400 total cells (keep control)
    print(f"Removing perturbations with < {MIN_CELLS_PER_PERT} total cells...")
    pert_counts = adata.obs[adata.obs['condition'] != 'control']['condition'].value_counts()
    eligible_perts = pert_counts[pert_counts >= MIN_CELLS_PER_PERT].index.tolist()
    n_removed = len(pert_counts) - len(eligible_perts)

    # Keep eligible perturbations + control
    keep_mask = adata.obs['condition'].isin(eligible_perts) | (adata.obs['condition'] == 'control')
    adata = adata[keep_mask].copy()

    print(f"  Kept {len(eligible_perts)} perturbations (removed {n_removed})")
    print(f"  Eligible perturbations: {eligible_perts[:10]}{'...' if len(eligible_perts) > 10 else ''}")

    # 10. Verify each eligible pert has enough cells in each half
    for pert in eligible_perts:
        fh = (adata.obs['condition'] == pert) & (adata.obs['tech_dup_split'] == 'first_half')
        sh = (adata.obs['condition'] == pert) & (adata.obs['tech_dup_split'] == 'second_half')
        n_fh, n_sh = fh.sum(), sh.sum()
        max_per_half = max(CELL_COUNTS) // 2  # 200
        if n_fh < max_per_half or n_sh < max_per_half:
            print(f"  WARNING: {pert} has {n_fh} first_half, {n_sh} second_half "
                  f"(need >= {max_per_half} per half for n_cells={max(CELL_COUNTS)})")

    # 11. Summary
    ctrl_count = (adata.obs['condition'] == 'control').sum()
    pert_counts_final = adata.obs[adata.obs['condition'] != 'control']['condition'].value_counts()
    print(f"\n--- SUMMARY ---")
    print(f"  Total cells: {adata.shape[0]}")
    print(f"  Total genes: {adata.shape[1]}")
    print(f"  Eligible perturbations: {len(eligible_perts)}")
    print(f"  Control cells: {ctrl_count}")
    print(f"  Cells per perturbation: mean={pert_counts_final.mean():.0f}, "
          f"min={pert_counts_final.min()}, max={pert_counts_final.max()}")
    print(f"  Cell counts to test: {CELL_COUNTS}")
    print()

    return adata


def process_single_cell_count(args):
    """Process a single cell count: subsample, normalize, HVG, DEGs, baselines.

    This function is designed to be called in parallel via ProcessPoolExecutor.

    Args:
        args: tuple of (adata_path_or_obj, n_cells, cache_dir, force)
              When running in parallel, adata is passed as a path to a temp file.

    Returns:
        Tuple of (n_cells, output_path, success, error_msg)
    """
    adata_input, n_cells, cache_dir, force = args

    output_path = Path(cache_dir) / f"norman19_n{n_cells}.h5ad"

    # Check cache
    if output_path.exists() and not force:
        print(f"[n={n_cells}] Cache exists at {output_path}, skipping")
        return (n_cells, str(output_path), True, None)

    try:
        t_start = time.time()

        # Load adata
        if isinstance(adata_input, (str, Path)):
            adata = sc.read_h5ad(str(adata_input))
        else:
            adata = adata_input.copy()

        # ------------------------------------------------------------------
        # Step 1: Subsample n_cells/2 from each half
        # ------------------------------------------------------------------
        np.random.seed(42 + n_cells)
        half_n = n_cells // 2

        non_ctrl_conditions = [c for c in adata.obs['condition'].unique()
                               if 'control' not in c.lower() and 'ctrl' not in c.lower()]

        cells_to_keep = []
        n_perts_sampled = 0

        for condition in non_ctrl_conditions:
            fh_cells = adata.obs[
                (adata.obs['condition'] == condition) &
                (adata.obs['tech_dup_split'] == 'first_half')
            ].index.tolist()
            sh_cells = adata.obs[
                (adata.obs['condition'] == condition) &
                (adata.obs['tech_dup_split'] == 'second_half')
            ].index.tolist()

            if len(fh_cells) < half_n or len(sh_cells) < half_n:
                print(f"[n={n_cells}] WARNING: {condition} has only {len(fh_cells)}/{len(sh_cells)} "
                      f"cells in first/second half, need {half_n}. Skipping.")
                continue

            sampled_fh = np.random.choice(fh_cells, size=half_n, replace=False)
            sampled_sh = np.random.choice(sh_cells, size=half_n, replace=False)
            cells_to_keep.extend(sampled_fh)
            cells_to_keep.extend(sampled_sh)
            n_perts_sampled += 1

        # Keep all control cells
        ctrl_cells = adata.obs[adata.obs['condition'] == 'control'].index.tolist()
        cells_to_keep.extend(ctrl_cells)

        adata = adata[cells_to_keep].copy()
        print(f"[n={n_cells}] Subsampled {n_perts_sampled} perturbations to {n_cells} cells each "
              f"({half_n} per half). Total cells: {adata.shape[0]}")

        # ------------------------------------------------------------------
        # Step 2: Normalize
        # ------------------------------------------------------------------
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"[n={n_cells}] Normalized and log-transformed. Shape: {adata.shape}")

        # ------------------------------------------------------------------
        # Step 3: HVG selection
        # ------------------------------------------------------------------
        sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, subset=False)

        # Get perturbation genes
        perts = adata.obs['condition'].unique()
        pert_genes = set()
        for p in perts:
            if 'control' not in p.lower() and 'ctrl' not in p.lower():
                for g in p.split('+'):
                    pert_genes.add(g)

        hvg_genes = set(adata.var_names[adata.var.highly_variable])
        genes_to_keep = list(hvg_genes | pert_genes)
        genes_to_keep = [g for g in genes_to_keep if g in adata.var_names]
        n_hvg = len(hvg_genes & set(adata.var_names))
        n_pert_genes = len(pert_genes & set(adata.var_names))

        adata = adata[:, genes_to_keep].copy()
        print(f"[n={n_cells}] Selected {len(genes_to_keep)} genes "
              f"({n_hvg} HVGs + {n_pert_genes} perturbation genes)")

        # ------------------------------------------------------------------
        # Step 4: Compute DEGs (both halves, excluding control)
        # ------------------------------------------------------------------
        non_ctrl_conditions = [c for c in adata.obs['condition'].unique()
                               if 'control' not in c.lower() and 'ctrl' not in c.lower()]

        # 4a) Second-half DEGs (for interpolation p-values)
        sh_mask = (adata.obs['tech_dup_split'] == 'second_half')
        adata_sh = adata[sh_mask].copy()
        # Filter to perts with >= 2 cells
        sh_pert_counts = adata_sh.obs['condition'].value_counts()
        valid_sh_perts = sh_pert_counts[sh_pert_counts >= 2].index
        valid_sh_perts = [p for p in valid_sh_perts
                          if 'control' not in p.lower() and 'ctrl' not in p.lower()]
        adata_sh_deg = adata_sh[adata_sh.obs['condition'].isin(valid_sh_perts)].copy()

        print(f"[n={n_cells}] Computing second-half DEGs for {len(valid_sh_perts)} perturbations "
              f"(control excluded)...")
        sc.tl.rank_genes_groups(adata_sh_deg, 'condition',
                                method='t-test_overestim_var', reference='rest')

        names_df_sh = pd.DataFrame(adata_sh_deg.uns["rank_genes_groups"]["names"])
        pvals_adj_df_sh = pd.DataFrame(adata_sh_deg.uns["rank_genes_groups"]["pvals_adj"])
        scores_df_sh = pd.DataFrame(adata_sh_deg.uns["rank_genes_groups"]["scores"])

        names_df_dict = {}
        pvals_adj_df_dict = {}
        scores_df_dict = {}
        for pert in names_df_sh.columns:
            if 'control' in pert.lower() or 'ctrl' in pert.lower():
                continue
            key = f'{DATASET_NAME}_{pert}'
            names_df_dict[key] = names_df_sh[pert].tolist()
            pvals_adj_df_dict[key] = pvals_adj_df_sh[pert].tolist()
            scores_df_dict[key] = scores_df_sh[pert].tolist()

        adata.uns['names_df_dict'] = names_df_dict
        adata.uns['pvals_adj_df_dict'] = pvals_adj_df_dict
        adata.uns['scores_df_dict'] = scores_df_dict
        print(f"[n={n_cells}] Computed second-half DEGs for {len(names_df_dict)} perturbations")

        # 4b) First-half DEGs (for GT metric weights)
        fh_mask = (adata.obs['tech_dup_split'] == 'first_half')
        adata_fh = adata[fh_mask].copy()
        fh_pert_counts = adata_fh.obs['condition'].value_counts()
        valid_fh_perts = fh_pert_counts[fh_pert_counts >= 2].index
        valid_fh_perts = [p for p in valid_fh_perts
                          if 'control' not in p.lower() and 'ctrl' not in p.lower()]
        adata_fh_deg = adata_fh[adata_fh.obs['condition'].isin(valid_fh_perts)].copy()

        print(f"[n={n_cells}] Computing first-half (GT) DEGs for {len(valid_fh_perts)} perturbations "
              f"(control excluded)...")
        sc.tl.rank_genes_groups(adata_fh_deg, 'condition',
                                method='t-test_overestim_var', reference='rest')

        names_df_fh = pd.DataFrame(adata_fh_deg.uns["rank_genes_groups"]["names"])
        pvals_adj_df_fh = pd.DataFrame(adata_fh_deg.uns["rank_genes_groups"]["pvals_adj"])
        scores_df_fh = pd.DataFrame(adata_fh_deg.uns["rank_genes_groups"]["scores"])

        names_df_dict_gt = {}
        pvals_adj_df_dict_gt = {}
        scores_df_dict_gt = {}
        for pert in names_df_fh.columns:
            if 'control' in pert.lower() or 'ctrl' in pert.lower():
                continue
            key = f'{DATASET_NAME}_{pert}'
            names_df_dict_gt[key] = names_df_fh[pert].tolist()
            pvals_adj_df_dict_gt[key] = pvals_adj_df_fh[pert].tolist()
            scores_df_dict_gt[key] = scores_df_fh[pert].tolist()

        adata.uns['names_df_dict_gt'] = names_df_dict_gt
        adata.uns['pvals_adj_df_dict_gt'] = pvals_adj_df_dict_gt
        adata.uns['scores_df_dict_gt'] = scores_df_dict_gt
        print(f"[n={n_cells}] Computed first-half (GT) DEGs for {len(names_df_dict_gt)} perturbations")

        # ------------------------------------------------------------------
        # Step 5: Compute baselines
        # ------------------------------------------------------------------
        var_names = adata.var_names.tolist()

        # Ground truth: mean of first_half cells per perturbation
        gt_df = pd.DataFrame(columns=var_names)
        for condition in non_ctrl_conditions:
            fh_cells = adata[
                (adata.obs['condition'] == condition) &
                (adata.obs['tech_dup_split'] == 'first_half')
            ]
            if len(fh_cells) > 0:
                mean_expr = fh_cells.X.mean(axis=0)
                if hasattr(mean_expr, 'A1'):
                    mean_expr = mean_expr.A1
                else:
                    mean_expr = np.asarray(mean_expr).flatten()
                gt_df.loc[f'{DATASET_NAME}_{condition}'] = mean_expr

        gt_df = gt_df.astype(float)
        adata.uns['technical_duplicate_first_half_baseline'] = gt_df
        print(f"[n={n_cells}] Ground truth baseline: {len(gt_df)} perturbations")

        # Technical duplicate: mean of second_half cells per perturbation
        td_df = pd.DataFrame(columns=var_names)
        for condition in non_ctrl_conditions:
            sh_cells = adata[
                (adata.obs['condition'] == condition) &
                (adata.obs['tech_dup_split'] == 'second_half')
            ]
            if len(sh_cells) > 0:
                mean_expr = sh_cells.X.mean(axis=0)
                if hasattr(mean_expr, 'A1'):
                    mean_expr = mean_expr.A1
                else:
                    mean_expr = np.asarray(mean_expr).flatten()
                td_df.loc[f'{DATASET_NAME}_{condition}'] = mean_expr

        td_df = td_df.astype(float)
        adata.uns['technical_duplicate_second_half_baseline'] = td_df
        print(f"[n={n_cells}] Technical duplicate baseline: {len(td_df)} perturbations")

        # Mean baseline: average of all non-control perturbation means
        all_pert_means = pd.DataFrame(columns=var_names)
        for condition in non_ctrl_conditions:
            cond_cells = adata[adata.obs['condition'] == condition]
            if len(cond_cells) > 0:
                mean_expr = cond_cells.X.mean(axis=0)
                if hasattr(mean_expr, 'A1'):
                    mean_expr = mean_expr.A1
                else:
                    mean_expr = np.asarray(mean_expr).flatten()
                all_pert_means.loc[condition] = mean_expr

        dataset_mean = all_pert_means.astype(float).mean(axis=0)
        mean_baseline_df = pd.DataFrame(
            [dataset_mean.values],
            index=[DATASET_NAME],
            columns=var_names
        ).astype(float)
        adata.uns['dataset_mean_baseline'] = mean_baseline_df
        print(f"[n={n_cells}] Mean baseline computed from {len(all_pert_means)} perturbation means")

        # Control baseline: mean of all control cells
        ctrl_cells = adata[adata.obs['condition'] == 'control']
        if len(ctrl_cells) > 0:
            ctrl_mean = ctrl_cells.X.mean(axis=0)
            if hasattr(ctrl_mean, 'A1'):
                ctrl_mean = ctrl_mean.A1
            else:
                ctrl_mean = np.asarray(ctrl_mean).flatten()
            ctrl_baseline_df = pd.DataFrame(
                [ctrl_mean],
                index=[DATASET_NAME],
                columns=var_names
            ).astype(float)
            adata.uns['ctrl_baseline'] = ctrl_baseline_df
            print(f"[n={n_cells}] Control baseline from {len(ctrl_cells)} control cells")

        # Interpolated duplicate: alpha * tech_dup + (1-alpha) * mean
        interp_df = pd.DataFrame(columns=var_names)
        mean_values = mean_baseline_df.loc[DATASET_NAME].values

        for condition_key in td_df.index:
            # condition_key is like "norman19_GENE1+GENE2"
            if condition_key not in pvals_adj_df_dict:
                # No p-values for this perturbation, use mean baseline
                interp_df.loc[condition_key] = mean_values
                continue

            if condition_key not in names_df_dict:
                interp_df.loc[condition_key] = mean_values
                continue

            pvals_list = pvals_adj_df_dict[condition_key]
            names_list = names_df_dict[condition_key]

            # Create pval series aligned to var_names
            pvals_series = pd.Series(pvals_list, index=names_list)
            pvals_ordered = pvals_series.reindex(var_names, fill_value=1.0)
            pvals = pvals_ordered.values.astype(float)

            # alpha = 1 - pval (higher alpha = more weight to tech duplicate)
            alphas = 1 - pvals
            alphas = np.nan_to_num(alphas, nan=0.0)

            interpolated_values = (
                alphas * td_df.loc[condition_key].values.astype(float) +
                (1 - alphas) * mean_values.astype(float)
            )
            interp_df.loc[condition_key] = interpolated_values

        interp_df = interp_df.astype(float)
        adata.uns['interpolated_duplicate_baseline'] = interp_df
        print(f"[n={n_cells}] Interpolated baseline: {len(interp_df)} perturbations")

        print(f"[n={n_cells}] Computed baselines: GT={len(gt_df)}, TD={len(td_df)}, "
              f"mean, ctrl, interp={len(interp_df)}")

        # ------------------------------------------------------------------
        # Step 6: Save
        # ------------------------------------------------------------------
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(str(output_path))
        elapsed = time.time() - t_start
        print(f"[n={n_cells}] Saved to {output_path} ({elapsed:.1f}s)")

        return (n_cells, str(output_path), True, None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (n_cells, str(output_path), False, str(e))


def run_stage1(force=False, workers=1):
    """Run Stage 1: Preprocessing + Baselines for all cell counts."""
    print("\n" + "=" * 60)
    print("STAGE 1: PREPROCESSING + BASELINES")
    print("=" * 60)

    # Check which cell counts need processing
    to_process = []
    for n_cells in CELL_COUNTS:
        output_path = CACHE_DIR / f"norman19_n{n_cells}.h5ad"
        if output_path.exists() and not force:
            print(f"  [n={n_cells}] Cache exists, skipping")
        else:
            to_process.append(n_cells)

    if not to_process:
        print("All cell counts already cached. Use --force to recompute.")
        return

    print(f"Processing cell counts: {to_process}")

    # Load and prepare raw data once
    adata = load_and_prepare_raw_data()

    # Save prepared data to a temp file for parallel workers
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = CACHE_DIR / "_prepared_raw.h5ad"
    print(f"Saving prepared data to temp file for parallel processing: {temp_path}")
    adata.write_h5ad(str(temp_path))

    # Build args
    args_list = [(str(temp_path), n_cells, str(CACHE_DIR), force)
                 for n_cells in to_process]

    if workers > 1 and len(to_process) > 1:
        print(f"\nRunning {len(to_process)} cell counts in parallel with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_cell_count, a): a[1]
                       for a in args_list}
            for future in as_completed(futures):
                n_cells = futures[future]
                result = future.result()
                if not result[2]:
                    print(f"  [n={n_cells}] FAILED: {result[3]}")
    else:
        print(f"\nRunning {len(to_process)} cell counts sequentially...")
        for a in args_list:
            result = process_single_cell_count(a)
            if not result[2]:
                print(f"  [n={a[1]}] FAILED: {result[3]}")

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
        print(f"Cleaned up temp file: {temp_path}")

    print("\nStage 1 complete.")


# ============================================================================
# Stage 2: Metrics + DRF
# ============================================================================

def compute_deg_weights(scores_dict, names_dict, var_names):
    """Compute DEG-based weights per perturbation.

    Replicates DataManager._precompute_deg_weights() logic:
    absolute scores -> min-max normalize -> square -> reindex to var_names.

    Args:
        scores_dict: Dict mapping cov_pert_key to score arrays
        names_dict: Dict mapping cov_pert_key to gene name arrays
        var_names: List of gene names to align to

    Returns:
        Dict mapping cov_pert_key to weight arrays aligned to var_names
    """
    weights_dict = {}
    for cov_pert_key in scores_dict:
        if 'control' in cov_pert_key.lower():
            continue

        scores = np.array(scores_dict[cov_pert_key])
        gene_names = names_dict[cov_pert_key]

        abs_scores = np.abs(scores)
        min_val = np.min(abs_scores)
        max_val = np.max(abs_scores)

        if max_val == min_val:
            normalized_weights = np.zeros_like(abs_scores)
        else:
            normalized_weights = (abs_scores - min_val) / (max_val - min_val)

        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
        normalized_weights = np.square(normalized_weights)

        weights_df = pd.DataFrame({'gene': gene_names, 'weight': normalized_weights})
        weights_aggregated = weights_df.groupby('gene')['weight'].max()
        weights = weights_aggregated.reindex(var_names, fill_value=0.0)

        weights_dict[cov_pert_key] = weights.values

    return weights_dict


def compute_deg_mask(pvals_dict, names_dict, var_names, topn=100):
    """Compute DEG mask (top N genes by p-value) per perturbation.

    Replicates DataManager.get_deg_mask(..., topn=100) logic.

    Args:
        pvals_dict: Dict mapping cov_pert_key to p-value arrays
        names_dict: Dict mapping cov_pert_key to gene name arrays
        var_names: List of gene names to align to
        topn: Number of top DEGs to select

    Returns:
        Dict mapping cov_pert_key to boolean mask arrays aligned to var_names
    """
    masks_dict = {}
    for cov_pert_key in pvals_dict:
        if 'control' in cov_pert_key.lower():
            continue

        pvals = np.array(pvals_dict[cov_pert_key])
        gene_names = names_dict[cov_pert_key]

        pvals_df = pd.DataFrame({'gene': gene_names, 'pval': pvals})

        # Select top N by p-value
        pvals_df_sorted = pvals_df.sort_values(by='pval', ascending=True).head(topn)
        pvals_df['significant'] = False
        pvals_df.loc[pvals_df.index.isin(pvals_df_sorted.index), 'significant'] = True

        deg_mask_aggregated = pvals_df.groupby('gene')['significant'].any()
        deg_mask = deg_mask_aggregated.reindex(var_names, fill_value=False)

        masks_dict[cov_pert_key] = deg_mask.values

    return masks_dict


def compute_nir(predictions_df, ground_truth_df):
    """Compute NIR (Nearest In-distribution Reference) scores.

    For each perturbation, measures the fraction of times its predicted profile
    is closer to its correct ground truth than to other perturbations' ground truths.

    Args:
        predictions_df: DataFrame with predicted expression (pert_key as index)
        ground_truth_df: DataFrame with ground truth expression (pert_key as index)

    Returns:
        Dict mapping pert_key to NIR score (0-1)
    """
    # Align to common keys
    common_keys = predictions_df.index.intersection(ground_truth_df.index)
    if len(common_keys) < 2:
        return {k: np.nan for k in common_keys}

    pred = predictions_df.loc[common_keys]
    truth = ground_truth_df.loc[common_keys]

    # Compute pairwise distance matrix
    distance_matrix = cdist(pred.values, truth.values, metric='euclidean')

    nir_scores = {}
    for i, pert_key in enumerate(common_keys):
        correct_distance = distance_matrix[i, i]
        comparisons = []
        for j in range(len(common_keys)):
            if i != j:
                comparisons.append(1 if correct_distance < distance_matrix[i, j] else 0)
        nir_scores[pert_key] = np.mean(comparisons) if comparisons else 0.0

    return nir_scores


def compute_metrics_for_cell_count(n_cells, cache_dir):
    """Compute all metrics for a single cell count.

    Args:
        n_cells: Number of cells per perturbation
        cache_dir: Path to cache directory

    Returns:
        List of dicts with columns: n_cells, baseline, metric, perturbation, value
    """
    h5ad_path = Path(cache_dir) / f"norman19_n{n_cells}.h5ad"
    if not h5ad_path.exists():
        print(f"[n={n_cells}] h5ad not found at {h5ad_path}, skipping")
        return []

    print(f"[n={n_cells}] Loading {h5ad_path}...")
    adata = sc.read_h5ad(str(h5ad_path))
    var_names = adata.var_names.tolist()

    # Extract baselines
    gt_df = adata.uns['technical_duplicate_first_half_baseline']
    td_df = adata.uns['technical_duplicate_second_half_baseline']
    mean_df = adata.uns['dataset_mean_baseline']
    ctrl_df = adata.uns['ctrl_baseline']
    interp_df = adata.uns['interpolated_duplicate_baseline']

    # Get vectors
    mean_vec = mean_df.loc[DATASET_NAME].values.astype(float)
    ctrl_vec = ctrl_df.loc[DATASET_NAME].values.astype(float)

    # Compute DEG weights from GT DEGs
    scores_dict_gt = adata.uns['scores_df_dict_gt']
    names_dict_gt = adata.uns['names_df_dict_gt']
    pvals_dict_gt = adata.uns['pvals_adj_df_dict_gt']

    print(f"[n={n_cells}] Computing DEG weights and masks...")
    weights_dict = compute_deg_weights(scores_dict_gt, names_dict_gt, var_names)
    masks_dict = compute_deg_mask(pvals_dict_gt, names_dict_gt, var_names, topn=100)
    print(f"[n={n_cells}] DEG weights for {len(weights_dict)} perturbations, "
          f"DEG masks for {len(masks_dict)} perturbations")

    # Common perturbation keys (present in all baselines)
    common_perts = sorted(
        set(gt_df.index) & set(td_df.index) & set(interp_df.index)
    )
    common_perts = [p for p in common_perts
                    if 'control' not in p.lower() and 'ctrl' not in p.lower()]
    print(f"[n={n_cells}] Computing metrics for {len(common_perts)} perturbations")

    # Build DataFrames for NIR computation
    baselines_dict = {
        'dataset_mean': pd.DataFrame(
            [mean_vec] * len(common_perts),
            index=common_perts,
            columns=var_names
        ),
        'technical_duplicate': td_df.loc[common_perts],
        'control': pd.DataFrame(
            [ctrl_vec] * len(common_perts),
            index=common_perts,
            columns=var_names
        ),
        'interpolated_duplicate': interp_df.loc[common_perts],
    }
    gt_common = gt_df.loc[common_perts]

    # Compute NIR for each baseline
    print(f"[n={n_cells}] Computing NIR scores...")
    nir_scores_by_baseline = {}
    for baseline_name, baseline_df in baselines_dict.items():
        nir_scores_by_baseline[baseline_name] = compute_nir(baseline_df, gt_common)

    # Compute per-perturbation metrics
    results = []

    for baseline_name, baseline_df in baselines_dict.items():
        metric_accum = {
            'mse': [], 'wmse': [], 'nir': [],
            'pearson_deltactrl': [], 'pearson_deltactrl_degs': [],
            'weighted_r2_deltactrl': []
        }

        for pert_key in common_perts:
            pred_vec = baseline_df.loc[pert_key].values.astype(float)
            gt_vec = gt_common.loc[pert_key].values.astype(float)

            # DEG weights and mask
            weights = weights_dict[pert_key] if pert_key in weights_dict else np.zeros(len(var_names))
            deg_mask = masks_dict[pert_key] if pert_key in masks_dict else np.zeros(len(var_names), dtype=bool)

            # MSE
            mse_val = mse(pred_vec, gt_vec)
            results.append({'n_cells': n_cells, 'baseline': baseline_name,
                            'metric': 'mse', 'perturbation': pert_key, 'value': mse_val})
            metric_accum['mse'].append(mse_val)

            # WMSE
            wmse_val = wmse(pred_vec, gt_vec, weights)
            results.append({'n_cells': n_cells, 'baseline': baseline_name,
                            'metric': 'wmse', 'perturbation': pert_key, 'value': wmse_val})
            metric_accum['wmse'].append(wmse_val)

            # NIR (already computed)
            nir_val = nir_scores_by_baseline[baseline_name].get(pert_key, np.nan)
            results.append({'n_cells': n_cells, 'baseline': baseline_name,
                            'metric': 'nir', 'perturbation': pert_key, 'value': nir_val})
            metric_accum['nir'].append(nir_val)

            # Delta control vectors
            pred_delta_ctrl = pred_vec - ctrl_vec
            gt_delta_ctrl = gt_vec - ctrl_vec

            # Pearson delta control (all genes)
            try:
                pdc_val, _ = pearsonr(pred_delta_ctrl, gt_delta_ctrl)
            except Exception:
                pdc_val = np.nan
            results.append({'n_cells': n_cells, 'baseline': baseline_name,
                            'metric': 'pearson_deltactrl', 'perturbation': pert_key,
                            'value': pdc_val})
            metric_accum['pearson_deltactrl'].append(pdc_val)

            # Pearson delta control DEGs (top 100)
            if deg_mask.sum() > 2:
                try:
                    pdc_deg_val, _ = pearsonr(pred_delta_ctrl[deg_mask],
                                              gt_delta_ctrl[deg_mask])
                except Exception:
                    pdc_deg_val = np.nan
            else:
                pdc_deg_val = np.nan
            results.append({'n_cells': n_cells, 'baseline': baseline_name,
                            'metric': 'pearson_deltactrl_degs', 'perturbation': pert_key,
                            'value': pdc_deg_val})
            metric_accum['pearson_deltactrl_degs'].append(pdc_deg_val)

            # Weighted R2 delta control
            wr2dc_val = r2_score_on_deltas(gt_delta_ctrl, pred_delta_ctrl, weights)
            results.append({'n_cells': n_cells, 'baseline': baseline_name,
                            'metric': 'weighted_r2_deltactrl', 'perturbation': pert_key,
                            'value': wr2dc_val})
            metric_accum['weighted_r2_deltactrl'].append(wr2dc_val)

        # Print summary for this baseline
        print(f"[n={n_cells}] {baseline_name}: "
              f"MSE={np.nanmean(metric_accum['mse']):.4f}, "
              f"WMSE={np.nanmean(metric_accum['wmse']):.4f}, "
              f"NIR={np.nanmean(metric_accum['nir']):.3f}, "
              f"rDC={np.nanmean(metric_accum['pearson_deltactrl']):.3f}, "
              f"rDC_DEG={np.nanmean(metric_accum['pearson_deltactrl_degs']):.3f}, "
              f"WR2DC={np.nanmean(metric_accum['weighted_r2_deltactrl']):.3f}")

    return results


def run_stage2(force=False, workers=1):
    """Run Stage 2: Metrics + DRF computation."""
    print("\n" + "=" * 60)
    print("STAGE 2: METRICS + DRF")
    print("=" * 60)

    if METRICS_CSV.exists() and not force:
        print(f"Metrics CSV already exists at {METRICS_CSV}. Use --force to recompute.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    if workers > 1 and len(CELL_COUNTS) > 1:
        print(f"Computing metrics for {len(CELL_COUNTS)} cell counts with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(compute_metrics_for_cell_count, n, str(CACHE_DIR)): n
                for n in CELL_COUNTS
            }
            for future in as_completed(futures):
                n_cells = futures[future]
                try:
                    result = future.result()
                    all_results.extend(result)
                except Exception as e:
                    print(f"[n={n_cells}] FAILED: {e}")
    else:
        print(f"Computing metrics for {len(CELL_COUNTS)} cell counts sequentially...")
        for n_cells in CELL_COUNTS:
            result = compute_metrics_for_cell_count(n_cells, str(CACHE_DIR))
            all_results.extend(result)

    if not all_results:
        print("No results to save!")
        return

    # Save long-format CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved per-perturbation metrics to {METRICS_CSV} ({len(results_df)} rows)")

    # Compute summary statistics
    summary = results_df.groupby(['n_cells', 'baseline', 'metric'])['value'].agg(
        ['mean', 'median', 'std', 'count']
    ).reset_index()
    summary.columns = ['n_cells', 'baseline', 'metric', 'mean', 'median', 'std', 'count']

    # Compute SEM
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary statistics to {SUMMARY_CSV} ({len(summary)} rows)")

    print("\nStage 2 complete.")


# ============================================================================
# Stage 3: Plotting
# ============================================================================

def run_stage3():
    """Run Stage 3: Generate plots from cached metrics."""
    print("\n" + "=" * 60)
    print("STAGE 3: PLOTTING")
    print("=" * 60)

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
    })

    if not SUMMARY_CSV.exists():
        print(f"Summary CSV not found at {SUMMARY_CSV}. Run stages 1+2 first.")
        return

    if not METRICS_CSV.exists():
        print(f"Metrics CSV not found at {METRICS_CSV}. Run stages 1+2 first.")
        return

    summary = pd.read_csv(SUMMARY_CSV)
    results_df = pd.read_csv(METRICS_CSV)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define metrics and their properties
    # Layout: top row = base metrics, bottom row = calibrated counterparts
    #   Col 0: MSE / WMSE
    #   Col 1: Pearson Delta Ctrl / Pearson Delta Ctrl DEGs
    #   Col 2: NIR / Weighted R² Delta Ctrl
    metrics_grid = [
        # Top row (base metrics)
        ('mse', {'title': 'MSE', 'lower_is_better': True}),
        ('pearson_deltactrl', {'title': 'Pearson Delta Control', 'lower_is_better': False}),
        ('nir', {'title': 'NIR', 'lower_is_better': False}),
        # Bottom row (calibrated counterparts)
        ('wmse', {'title': 'WMSE', 'lower_is_better': True}),
        ('pearson_deltactrl_degs', {'title': 'Pearson Delta Control (DEGs)', 'lower_is_better': False}),
        ('weighted_r2_deltactrl', {'title': 'Weighted R² Delta Control', 'lower_is_better': False}),
    ]
    metrics_config = dict(metrics_grid)

    baseline_colors = {
        'dataset_mean': '#1f77b4',
        'technical_duplicate': '#2ca02c',
        'control': '#d62728',
        'interpolated_duplicate': '#9467bd',
    }
    baseline_labels = {
        'dataset_mean': 'Dataset Mean',
        'technical_duplicate': 'Technical Duplicate',
        'control': 'Control',
        'interpolated_duplicate': 'Interpolated Duplicate',
    }

    # --- Metrics plot (2x3 panels) with line + SEM ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    baseline_order = ['dataset_mean', 'technical_duplicate', 'control',
                      'interpolated_duplicate']

    for idx, (metric_name, config) in enumerate(metrics_grid):
        ax = axes[idx]
        metric_data = summary[summary['metric'] == metric_name]

        for baseline_name in baseline_order:
            bl_data = metric_data[metric_data['baseline'] == baseline_name].sort_values('n_cells')
            if len(bl_data) == 0:
                continue

            ax.errorbar(
                bl_data['n_cells'], bl_data['mean'], yerr=bl_data['sem'],
                label=baseline_labels[baseline_name],
                color=baseline_colors[baseline_name],
                marker='o', capsize=3, linewidth=2, markersize=6
            )

        ax.set_xlabel('Cells per Perturbation')
        ax.set_ylabel(config['title'])
        ax.set_title(config['title'])
        ax.legend()
        ax.set_xscale('log')
        ax.set_xticks(CELL_COUNTS)
        ax.set_xticklabels([str(n) for n in CELL_COUNTS])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cell Count Sensitivity Analysis - Norman19', y=1.02)
    plt.tight_layout()

    metrics_plot_path = RESULTS_DIR / "cell_count_sensitivity_metrics.png"
    fig.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved metrics plot to {metrics_plot_path}")

    # --- DRF plot (interpolated duplicate vs dataset mean) with violins + jitter ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    sorted_counts = sorted(CELL_COUNTS)
    pos_map = {n: i for i, n in enumerate(sorted_counts)}

    # First pass: compute DRF data for all metrics to find global min
    all_drf_dfs = {}
    global_drf_min = 0.0
    for idx, (metric_name, config) in enumerate(metrics_grid):
        lower_is_better = config['lower_is_better']
        drf_data = []
        for n_cells in sorted_counts:
            mask_n = results_df['n_cells'] == n_cells
            mask_metric = results_df['metric'] == metric_name
            interp_vals = results_df[mask_n & mask_metric &
                                     (results_df['baseline'] == 'interpolated_duplicate')]
            mean_vals = results_df[mask_n & mask_metric &
                                   (results_df['baseline'] == 'dataset_mean')]
            if len(interp_vals) == 0 or len(mean_vals) == 0:
                continue
            merged = interp_vals[['perturbation', 'value']].merge(
                mean_vals[['perturbation', 'value']],
                on='perturbation', suffixes=('_interp', '_mean'))
            for _, row in merged.iterrows():
                interp_v = row['value_interp']
                mean_v = row['value_mean']
                if lower_is_better:
                    drf = (mean_v - interp_v) / mean_v if mean_v != 0 else np.nan
                else:
                    drf = (interp_v - mean_v) / (1 - mean_v) if (1 - mean_v) != 0 else np.nan
                drf_data.append({'n_cells': n_cells, 'drf': drf})
        if drf_data:
            drf_df = pd.DataFrame(drf_data).dropna(subset=['drf'])
            all_drf_dfs[metric_name] = drf_df
            global_drf_min = min(global_drf_min, drf_df['drf'].min())

    DRF_CLIP = -0.25

    # Print per-perturbation DRF summary
    print("\n--- DRF SUMMARY (per-perturbation, Interpolated Dup. vs Dataset Mean) ---")
    for metric_name, config in metrics_grid:
        if metric_name not in all_drf_dfs:
            continue
        drf_df = all_drf_dfs[metric_name]
        print(f"\n  {config['title']}:")
        for n_cells in sorted_counts:
            pts = drf_df[drf_df['n_cells'] == n_cells]['drf'].dropna()
            if len(pts) == 0:
                continue
            print(f"    n={n_cells:>3d}: median={pts.median():.3f}, "
                  f"mean={pts.mean():.3f}, std={pts.std():.3f}, n={len(pts)}")

    # Second pass: plot
    for idx, (metric_name, config) in enumerate(metrics_grid):
        ax = axes[idx]
        if metric_name not in all_drf_dfs:
            continue
        drf_df = all_drf_dfs[metric_name]

        # Clip DRF values for violin plotting
        drf_df_clipped = drf_df.copy()
        drf_df_clipped['drf_clipped'] = drf_df_clipped['drf'].clip(lower=DRF_CLIP)
        drf_df_clipped['is_clipped'] = drf_df_clipped['drf'] < DRF_CLIP

        # Build list of arrays for violin plot (using clipped values)
        violin_data = []
        violin_positions = []
        for n_cells in sorted_counts:
            pts = drf_df_clipped[drf_df_clipped['n_cells'] == n_cells]['drf_clipped'].values
            if len(pts) >= 2:
                violin_data.append(pts)
                violin_positions.append(pos_map[n_cells])

        if violin_data:
            vp = ax.violinplot(violin_data, positions=violin_positions,
                               showmeans=False, showmedians=False, showextrema=False,
                               widths=0.7)
            for body in vp['bodies']:
                body.set_facecolor('#9467bd')
                body.set_alpha(0.25)
                body.set_edgecolor('#9467bd')
                body.set_linewidth(0.5)

        rng = np.random.default_rng(42)
        for n_cells in sorted_counts:
            subset = drf_df_clipped[drf_df_clipped['n_cells'] == n_cells]
            if len(subset) == 0:
                continue
            x_jittered = pos_map[n_cells] + rng.uniform(-0.2, 0.2, len(subset))

            # Normal points (not clipped)
            normal = ~subset['is_clipped'].values
            if normal.any():
                ax.scatter(x_jittered[normal], subset['drf_clipped'].values[normal],
                           color='#9467bd', alpha=0.3, s=10,
                           edgecolors='none', rasterized=True, zorder=3)
            # Clipped points (down arrow)
            clipped = subset['is_clipped'].values
            if clipped.any():
                ax.scatter(x_jittered[clipped], subset['drf_clipped'].values[clipped],
                           color='#9467bd', alpha=0.5, s=30,
                           marker='v', edgecolors='none', rasterized=True, zorder=4)

        medians = drf_df.groupby('n_cells')['drf'].median().sort_index()
        median_positions = [pos_map[n] for n in medians.index]
        median_vals = medians.values.clip(min=DRF_CLIP)
        ax.scatter(median_positions, median_vals,
                   color='white', edgecolors='#7b4ea3', s=50, zorder=6,
                   linewidths=1.5, marker='D')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Cells per Perturbation')
        ax.set_ylabel('DRF')
        ax.set_title(config['title'])
        ax.set_ylim(DRF_CLIP - 0.05, 1.05)
        ax.set_xticks(range(len(sorted_counts)))
        ax.set_xticklabels([str(n) for n in sorted_counts])
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DRF vs Cell Count (Norman19)', y=1.02)
    plt.tight_layout()

    drf_plot_path = RESULTS_DIR / "cell_count_sensitivity_drf.png"
    fig.savefig(drf_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved DRF plot to {drf_plot_path}")

    print("\nStage 3 complete.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cell Count Sensitivity Analysis for Norman19"
    )
    parser.add_argument('--force', action='store_true',
                        help='Force recompute all stages')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only re-run plotting (stages 1+2 use cache)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    t_total = time.time()

    if args.plot_only:
        run_stage3()
    else:
        run_stage1(force=args.force, workers=args.workers)
        run_stage2(force=args.force, workers=args.workers)
        run_stage3()

    elapsed = time.time() - t_total
    print(f"\nTotal elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
