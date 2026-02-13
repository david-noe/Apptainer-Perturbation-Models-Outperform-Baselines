"""
Metrics calculation engine for CellSimBench framework.

Provides comprehensive metrics computation for evaluating perturbation
response predictions against ground truth.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import scanpy as sc
import warnings
from tqdm import tqdm

# Import metrics functions from data_manager
from .data_manager import mse, wmse, pearson, r2_score_on_deltas, DataManager, mae, wmae


def _gsea_worker(args):
    """Worker function for parallel GSEA pathway recovery computation.
    
    Must be module-level (not a method) for ProcessPoolExecutor pickling.
    Computes GSEA NES for both ground truth and predicted deltas, then
    returns the Pearson correlation between their NES vectors.
    """
    pred_deltas, truth_deltas, gene_names, hallmark_library = args

    # All-zero deltas (e.g. dataset_mean baseline vs itself) produce a singular
    # covariance matrix in blitzgsea's KDE step -- return NaN immediately.
    if np.all(pred_deltas == 0) or np.all(truth_deltas == 0):
        return np.nan

    import blitzgsea as blitz

    def run_gsea(deltas):
        signature = pd.DataFrame({0: gene_names, 1: deltas})
        result = blitz.gsea(
            signature, hallmark_library,
            permutations=500, shared_null=True,
            processes=2, seed=0,
        )
        return result['nes']

    gt_nes = run_gsea(truth_deltas)
    pred_nes = run_gsea(pred_deltas)

    common = gt_nes.index.intersection(pred_nes.index)
    if len(common) < 2:
        return np.nan
    corr, _ = pearsonr(pred_nes[common].values, gt_nes[common].values)
    return corr


class MetricsEngine:

    def __init__(self, data_manager: DataManager, run_nir: bool = False, run_gsea: bool = False, run_knn_graph: bool = False) -> None:
        """Initialize MetricsEngine with a DataManager instance.
        
        Args:
            data_manager: DataManager for accessing ground truth data and DEG weights.
            run_nir: Whether to run nir calculation (default False).
            run_gsea: Whether to run GSEA pathway recovery analysis (default False).
            run_knn_graph: Whether to run KNN graph Jaccard similarity analysis (default False).
        """
        self.data_manager = data_manager
        self.run_nir = run_nir
        self.run_gsea = run_gsea
        self.run_knn_graph = run_knn_graph
        self._hallmark_library = self._load_hallmark_library()
        
    def calculate_all_metrics(
        self,
        predictions: pd.DataFrame,
        predictions_deltas: Dict[str, pd.DataFrame],
        ground_truth: pd.DataFrame,
        ground_truth_deltas: Dict[str, pd.DataFrame],
        cached_nir_scores: Optional[Dict[str, float]] = None,
        cached_pds_scores: Optional[Dict[str, float]] = None,
        cached_gsea_scores: Optional[Dict[str, float]] = None,
        cached_knn_jaccard_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:

        # Ensure predictions and ground truth have the same var_names
        # Use sorted list to ensure deterministic, reproducible ordering
        common_var_names = sorted(set(predictions.columns) & set(ground_truth.columns))

        if not common_var_names:
            raise ValueError("Predictions and ground truth have different var_names")
        predictions = predictions[common_var_names]
        ground_truth = ground_truth[common_var_names]
        predictions_deltas = {key: df[common_var_names] for key, df in predictions_deltas.items()}
        ground_truth_deltas = {key: df[common_var_names] for key, df in ground_truth_deltas.items()}


        # Calculate nir scores (needs full dataset) - only if enabled
        if cached_nir_scores is not None:
            # Use cached scores
            nir_scores = cached_nir_scores
            pds_scores = cached_pds_scores
        elif self.run_nir:
            # Calculate fresh scores
            nir_scores = self._calculate_nir_scores(
                predictions, ground_truth
            )
            # PDS is NIR using Manhattan distance
            pds_scores = self._calculate_nir_scores(
                predictions, ground_truth, metric = "cityblock"
            )
        else:
            # Skip nir analysis, provide default scores
            nir_scores = {key: 0.0 for key in predictions.index}
            pds_scores = {key: 0.0 for key in predictions.index}

        # Calculate KNN graph Jaccard scores (needs full dataset) - only if enabled
        if cached_knn_jaccard_scores is not None:
            knn_jaccard_scores = cached_knn_jaccard_scores
        elif self.run_knn_graph:
            knn_jaccard_scores = self._calculate_knn_jaccard_scores(
                predictions_deltas['deltamean'], ground_truth_deltas['deltamean']
            )
        else:
            knn_jaccard_scores = {key: 0.0 for key in predictions.index}

        # Get all covariate-condition pairs from DataFrame index
        cov_condition_pairs = [(key.split('_')[0], '_'.join(key.split('_')[1:])) 
                              for key in predictions.index]
        
        # Calculate metrics for each covariate-condition pair
        condition_metrics = {}

        for covariate_value, condition in tqdm(cov_condition_pairs):
            cov_pert_key = f"{covariate_value}_{condition}"

            pred_expression = predictions.loc[cov_pert_key].values

            if cov_pert_key not in ground_truth.index:
                print(f"Covariate-condition pair {cov_pert_key} not found in ground truth")
                continue
            truth_expression = ground_truth.loc[cov_pert_key].values
            
            # Get pre-computed deltas
            pred_deltas_ctrl = predictions_deltas['deltactrl'].loc[cov_pert_key].values
            truth_deltas_ctrl = ground_truth_deltas['deltactrl'].loc[cov_pert_key].values
            pred_deltas_mean = predictions_deltas['deltamean'].loc[cov_pert_key].values
            truth_deltas_mean = ground_truth_deltas['deltamean'].loc[cov_pert_key].values


            # Get DEG weights and mask using covariate and perturbation
            weights = self.data_manager.get_deg_weights(covariate_value, condition, gene_order=common_var_names)
            deg_mask = self.data_manager.get_deg_mask(covariate_value, condition, gene_order=common_var_names, topn=100)
            deg_directions = self.data_manager.get_deg_directions(covariate_value, condition, gene_order=common_var_names)
            condition_metrics[cov_pert_key] = {
                'mse': self._calculate_mse(pred_expression, truth_expression),
                'mse_degs': self._calculate_mse(pred_expression[deg_mask], truth_expression[deg_mask]),
                'mae': self._calculate_mae(pred_expression, truth_expression),
                'mae_degs': self._calculate_mae(pred_expression[deg_mask], truth_expression[deg_mask]),
                'wmse': self._calculate_wmse(pred_expression, truth_expression, weights),
                'wmae': self._calculate_wmae(pred_expression, truth_expression, weights),
                
                # Delta metrics with control baseline - use pre-supplied deltas
                'pearson_deltactrl': self._calculate_pearson_delta_direct(pred_deltas_ctrl, truth_deltas_ctrl),
                'pearson_deltactrl_degs': self._calculate_pearson_delta_direct(
                    pred_deltas_ctrl[deg_mask], truth_deltas_ctrl[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'r2_deltactrl': self._calculate_r2_delta_direct(pred_deltas_ctrl, truth_deltas_ctrl),
                'r2_deltactrl_degs': self._calculate_r2_delta_direct(
                    pred_deltas_ctrl[deg_mask], truth_deltas_ctrl[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'weighted_r2_deltactrl': self._calculate_weighted_r2_delta_direct(
                    pred_deltas_ctrl, truth_deltas_ctrl, weights
                ),
                
                # Delta metrics with dataset mean baseline - use pre-supplied deltas
                'pearson_deltapert': self._calculate_pearson_delta_direct(pred_deltas_mean, truth_deltas_mean),
                'pearson_deltapert_degs': self._calculate_pearson_delta_direct(
                    pred_deltas_mean[deg_mask], truth_deltas_mean[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'r2_deltapert': self._calculate_r2_delta_direct(pred_deltas_mean, truth_deltas_mean),
                'r2_deltapert_degs': self._calculate_r2_delta_direct(
                    pred_deltas_mean[deg_mask], truth_deltas_mean[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'weighted_r2_deltapert': self._calculate_weighted_r2_delta_direct(
                    pred_deltas_mean, truth_deltas_mean, weights
                ),
                # nir metrics
                'nir': nir_scores[cov_pert_key] if cov_pert_key in nir_scores else np.nan,
                'pds': pds_scores[cov_pert_key] if cov_pert_key in pds_scores else np.nan,

                # KNN graph Jaccard similarity metric
                'knn_jaccard_deltapert': knn_jaccard_scores[cov_pert_key] if cov_pert_key in knn_jaccard_scores else np.nan,

                # # DEG direction recovery metric (only vs dataset mean / pert baseline)
                # 'deg_recovery_deltapert': self._calculate_deg_recovery(pred_deltas_mean, deg_directions, deg_mask),
            }

        # Phase 2: GSEA pathway recovery (use cached scores, compute in parallel, or skip)
        if cached_gsea_scores is not None:
            for key in condition_metrics:
                condition_metrics[key]['pathway_recovery_deltapert'] = cached_gsea_scores[key] if key in cached_gsea_scores else np.nan
        elif self.run_gsea:
            from concurrent.futures import ProcessPoolExecutor
            gsea_args = []
            gsea_keys = []
            for cov_pert_key in condition_metrics:
                pred_deltas_mean = predictions_deltas['deltamean'].loc[cov_pert_key].values
                truth_deltas_mean = ground_truth_deltas['deltamean'].loc[cov_pert_key].values
                gsea_args.append((pred_deltas_mean, truth_deltas_mean, common_var_names, self._hallmark_library))
                gsea_keys.append(cov_pert_key)

            with ProcessPoolExecutor(max_workers=36) as executor:
                gsea_results = list(tqdm(
                    executor.map(_gsea_worker, gsea_args),
                    total=len(gsea_args),
                    desc="GSEA pathway recovery"
                ))

            for key, score in zip(gsea_keys, gsea_results):
                condition_metrics[key]['pathway_recovery_deltapert'] = score
        else:
            for key in condition_metrics:
                condition_metrics[key]['pathway_recovery_deltapert'] = 0.0
            
        # Reorganize to metric -> cov_pert_key -> score format
        organized_metrics = {}
        all_metrics = condition_metrics[cov_pert_key].keys()
        for metric in all_metrics:
            organized_metrics[metric] = {
                cov_pert_key: condition_metrics[cov_pert_key][metric] 
                for cov_pert_key in condition_metrics.keys() if metric in condition_metrics[cov_pert_key].keys()
            }
        
        return organized_metrics
    
    def _calculate_mse(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Calculate MSE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            
        Returns:
            Mean squared error.
        """
        return mse(pred, truth)

    def _calculate_mae(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Calculate MAE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            
        Returns:
            Mean absolute error.
        """
        return mae(pred, truth)
    
    def _calculate_wmae(self, pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted MAE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted mean absolute error.
        """
        return wmae(pred, truth, weights)
    
    def _calculate_wmse(self, pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted MSE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted mean squared error.
        """
        return wmse(pred, truth, weights)
    
    def _calculate_pearson_delta(self, pred: np.ndarray, truth: np.ndarray, 
                               control: np.ndarray) -> float:
        """Calculate Pearson correlation of deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            
        Returns:
            Pearson correlation coefficient of deltas from control.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        try:
            corr, _ = pearsonr(delta_pred, delta_truth)
            return corr
        except:
            return np.nan
    
    def _calculate_pearson_delta_degs(self, pred: np.ndarray, truth: np.ndarray,
                                    control: np.ndarray, deg_mask: np.ndarray) -> float:
        """Calculate Pearson correlation of deltas for DEGs only.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            deg_mask: Boolean mask indicating DEG positions.
            
        Returns:
            Pearson correlation coefficient for DEGs only.
        """
        delta_pred = pred[deg_mask] - control[deg_mask]
        delta_truth = truth[deg_mask] - control[deg_mask]
        try:
            corr, _ = pearsonr(delta_pred, delta_truth)
            return corr
        except:
            return np.nan
    
    
    def _calculate_pearson_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray) -> float:
        """Calculate Pearson correlation on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            
        Returns:
            Pearson correlation coefficient of deltas.
        """
        corr, _ = pearsonr(pred_deltas, truth_deltas)
        return corr
    
    def _calculate_r2_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray) -> float:
        """Calculate R² on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            
        Returns:
            R² score on pre-computed delta values.
        """
        return r2_score_on_deltas(truth_deltas, pred_deltas)
    
    def _calculate_weighted_r2_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray, 
                                          weights: np.ndarray) -> float:
        """Calculate weighted R² on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted R² score on pre-computed delta values.
        """
        return r2_score_on_deltas(truth_deltas, pred_deltas, weights)
    
    def _calculate_deg_recovery(self, pred_deltas: np.ndarray, deg_directions: np.ndarray, 
                                deg_mask: np.ndarray) -> float:
        """Calculate DEG direction recovery score.
        
        For each significant DEG, checks whether the sign of the predicted delta
        matches the ground-truth DEG direction. Returns the fraction of significant
        DEGs with correctly recovered direction.
        
        Args:
            pred_deltas: Pre-computed predicted delta values (aligned to gene_order).
            deg_directions: Array of DEG directions (+1/-1/0, aligned to gene_order).
            deg_mask: Boolean mask of significant DEGs (aligned to gene_order).
            
        Returns:
            Fraction of significant DEGs with correct direction in predicted deltas.
        """
        if deg_mask.sum() == 0:
            return np.nan
        pred_signs = np.sign(pred_deltas[deg_mask])
        gt_signs = deg_directions[deg_mask]
        return np.mean(pred_signs == gt_signs)
    
    def _load_hallmark_library(self) -> Dict[str, List[str]]:
        """Load MSigDB Hallmark 2020 gene set library from Enrichr GMT format.
        
        Returns:
            Dictionary mapping pathway names to lists of gene symbols.
        """
        hallmark_path = Path(__file__).parent.parent.parent / 'data' / 'ref' / 'MSigDB_Hallmark_2020.txt'
        library = {}
        with open(hallmark_path) as f:
            for line in f:
                if line.strip():
                    items = line.strip().split('\t')
                    pathway_name = items[0]
                    genes = [g for g in items[2:] if g]
                    library[pathway_name] = genes
        return library
    
    def _calculate_nir_scores(
        self, 
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        metric='euclidean'
    ) -> Dict[str, float]:
        """Calculate nir for all perturbations within their covariate groups.
        
        For each perturbation, measures the fraction of times its predicted profile
        is closer to its correct ground truth than to other perturbations' ground truths
        WITHIN THE SAME COVARIATE GROUP.
        
        Args:
            predictions: DataFrame with predicted expression profiles (cov_pert_key as index)
            ground_truth: DataFrame with ground truth expression profiles (cov_pert_key as index)
            metric: metric to use for scipy cdist (default: "euclidean")
            
        Returns:
            Dict mapping cov_pert_key to nir score (0-1)
        """
        from scipy.spatial.distance import cdist
        
        nir_scores = {}
        
        # Group perturbations by covariate
        covariate_groups = {}
        for pert_key in predictions.index:
            covariate = pert_key.split('_')[0]
            if covariate not in covariate_groups:
                covariate_groups[covariate] = []
            covariate_groups[covariate].append(pert_key)
        
        # Calculate nir within each covariate group
        for covariate, pert_keys in covariate_groups.items():
            
            # Filter to only perturbations present in both predictions and ground truth
            valid_pert_keys = [pk for pk in pert_keys if pk in ground_truth.index]
            missing_pert_keys = [pk for pk in pert_keys if pk not in ground_truth.index]
            
            if missing_pert_keys:
                print(f"Warning: {len(missing_pert_keys)} perturbations not in ground truth for covariate {covariate}, skipping those")
            
            if len(valid_pert_keys) < 2:
                # Need at least 2 perturbations to calculate nir
                print(f"Skipping covariate {covariate}: only {len(valid_pert_keys)} valid perturbations (need ≥2)")
                continue
            
            # Get predictions and ground truths for valid perturbations only
            predictions_cov = predictions.loc[valid_pert_keys]
            ground_truth_cov = ground_truth.loc[valid_pert_keys]
            pert_keys = valid_pert_keys  # Use only valid keys for the rest of the calculation
            
            # Compute pairwise distance matrix for this covariate group
            distance_matrix = cdist(
                predictions_cov.values, 
                ground_truth_cov.values, 
                metric=metric
            )
            
            # Calculate nir for each perturbation in this covariate
            for i, pert_key in tqdm(enumerate(pert_keys), desc="Calculating nir for covariate " + covariate):
                # Distance from this prediction to its correct ground truth
                correct_distance = distance_matrix[i, i]
                
                # Compare to all OTHER ground truths within same covariate
                comparisons = []
                for j in range(len(pert_keys)):
                    if i != j:  # Skip self-comparison
                        # Is prediction closer to correct GT than to this other GT?
                        comparisons.append(1 if correct_distance < distance_matrix[i, j] else 0)
                
                # Average across all comparisons
                nir_scores[pert_key] = np.mean(comparisons) if comparisons else 0.0
        return nir_scores 

    def _calculate_knn_jaccard_scores(
        self,
        predictions_deltas_mean: pd.DataFrame,
        ground_truth_deltas_mean: pd.DataFrame,
        k: int = 20,
    ) -> Dict[str, float]:
        """Calculate KNN graph Jaccard similarity for all perturbations globally.
        
        Builds a K-nearest-neighbor graph on ground-truth delta-mean profiles and
        another on predicted delta-mean profiles (across all perturbations globally).
        For each perturbation, computes the Jaccard similarity between its neighbor
        sets in the two graphs.
        
        Args:
            predictions_deltas_mean: DataFrame of predicted delta-from-mean profiles
                (cov_pert_key as index, genes as columns).
            ground_truth_deltas_mean: DataFrame of ground-truth delta-from-mean profiles
                (cov_pert_key as index, genes as columns).
            k: Number of nearest neighbors (default 20).
            
        Returns:
            Dict mapping cov_pert_key to Jaccard similarity score (0-1).
        """
        from scipy.spatial.distance import cdist

        # Align to common perturbation keys
        common_keys = predictions_deltas_mean.index.intersection(ground_truth_deltas_mean.index)
        if len(common_keys) < 2:
            print(f"KNN Jaccard: fewer than 2 common perturbation keys, skipping")
            return {}

        pred = predictions_deltas_mean.loc[common_keys]
        truth = ground_truth_deltas_mean.loc[common_keys]

        n = len(common_keys)
        effective_k = min(k, n - 1)

        # Compute pairwise distance matrices (each perturbation vs all others)
        pred_dist = cdist(pred.values, pred.values, metric='euclidean')
        truth_dist = cdist(truth.values, truth.values, metric='euclidean')

        # For each perturbation, find K nearest neighbors (excluding self)
        # argsort gives indices sorted by distance; index 0 is self (distance 0), so take [1:k+1]
        pred_knn = np.argsort(pred_dist, axis=1)[:, 1:effective_k + 1]
        truth_knn = np.argsort(truth_dist, axis=1)[:, 1:effective_k + 1]

        # Compute per-perturbation Jaccard similarity of neighbor sets
        knn_jaccard_scores = {}
        for i, pert_key in tqdm(enumerate(common_keys), total=n, desc="Calculating KNN Jaccard"):
            pred_neighbors = set(pred_knn[i])
            truth_neighbors = set(truth_knn[i])
            intersection = len(pred_neighbors & truth_neighbors)
            union = len(pred_neighbors | truth_neighbors)
            knn_jaccard_scores[pert_key] = intersection / union if union > 0 else 0.0

        return knn_jaccard_scores