"""
Utility functions for CellFlow CellSimBench integration.
"""

import logging
from typing import Dict, Tuple, Set

import numpy as np
import pandas as pd
import scanpy as sc

log = logging.getLogger(__name__)


def split_condition(cond: str) -> Tuple[str, str]:
    """Split a CellSimBench condition string into gene_target_1 and gene_target_2.
    
    Args:
        cond: Condition string, e.g. "AHR", "AHR+FEV", or "control".
        
    Returns:
        Tuple of (gene_target_1, gene_target_2).
    """
    if cond == "control":
        return "control", "control"
    if "+" in cond:
        parts = cond.split("+")
        return parts[0], parts[1]
    return cond, "control"


def prepare_gene_embeddings(adata: sc.AnnData, embedding_key: str) -> Dict[str, np.ndarray]:
    """Convert ESM2 embeddings from DataFrame in adata.uns to dict format for CellFlow.
    
    Args:
        adata: AnnData object with embeddings in adata.uns[embedding_key] as a DataFrame.
        embedding_key: Key in adata.uns containing the embedding DataFrame.
        
    Returns:
        Dict mapping gene names to embedding numpy arrays, including a "control" zero token.
    """
    esm2_df = adata.uns[embedding_key]
    gene_embeddings = {gene: np.array(esm2_df.loc[gene]) for gene in esm2_df.index}
    # Add control token (zeros with same dimension)
    gene_embeddings["control"] = np.zeros(esm2_df.shape[1])
    
    log.info(f"Prepared {len(gene_embeddings)} gene embeddings (including control token)")
    log.info(f"Embedding dimension: {esm2_df.shape[1]}")
    
    return gene_embeddings


def filter_cells_by_embedding_availability(
    adata: sc.AnnData,
    gene_embeddings: Dict[str, np.ndarray]
) -> sc.AnnData:
    """Filter AnnData to only keep cells where both gene targets have embeddings.
    
    Args:
        adata: AnnData with gene_target_1 and gene_target_2 columns in obs.
        gene_embeddings: Dict of gene name -> embedding array.
        
    Returns:
        Filtered AnnData copy.
    """
    available_genes = set(gene_embeddings.keys())
    
    # All perturbation genes (excluding control)
    all_pert_genes = set(adata.obs.gene_target_1.unique()) | set(adata.obs.gene_target_2.unique())
    all_pert_genes.discard("control")
    
    genes_with_embeddings = all_pert_genes & available_genes
    missing_genes = all_pert_genes - available_genes
    
    log.info(f"Total unique perturbation genes: {len(all_pert_genes)}")
    log.info(f"Genes with embeddings: {len(genes_with_embeddings)}")
    log.info(f"Genes missing embeddings: {len(missing_genes)}")
    if missing_genes:
        log.info(f"Missing genes: {sorted(missing_genes)[:20]}...")
    
    # Filter: keep cells where both targets have embeddings (or are "control")
    valid_genes = genes_with_embeddings | {"control"}
    mask = (
        adata.obs.gene_target_1.isin(valid_genes) &
        adata.obs.gene_target_2.isin(valid_genes)
    )
    
    n_before = len(adata)
    adata_filtered = adata[mask].copy()
    n_after = len(adata_filtered)
    log.info(f"Filtered from {n_before} to {n_after} cells ({n_before - n_after} removed)")
    
    return adata_filtered


def prepare_cellflow_obs(adata: sc.AnnData) -> sc.AnnData:
    """Add CellFlow-required obs columns to AnnData.
    
    Adds:
        - is_control: boolean flag
        - gene_target_1, gene_target_2: split condition into targets
        
    Args:
        adata: AnnData with 'condition' column in obs.
        
    Returns:
        AnnData with additional obs columns (modified in-place, also returned).
    """
    # Create is_control boolean column
    adata.obs["is_control"] = adata.obs.condition.astype(str).isin(["control", "ctrl", "ctrl_iegfp", ""])
    
    # Split condition into gene_target_1 and gene_target_2
    condition_str = adata.obs.condition.astype(str)
    splits = [split_condition(c) for c in condition_str]
    adata.obs["gene_target_1"] = [s[0] for s in splits]
    adata.obs["gene_target_2"] = [s[1] for s in splits]
    
    log.info(f"Prepared CellFlow obs columns:")
    log.info(f"  Control cells: {adata.obs.is_control.sum()}")
    log.info(f"  Perturbed cells: {(~adata.obs.is_control).sum()}")
    log.info(f"  Unique conditions: {adata.obs.condition.nunique()}")
    
    return adata
