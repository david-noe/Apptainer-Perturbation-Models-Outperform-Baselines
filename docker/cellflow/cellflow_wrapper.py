"""
CellFlow model wrapper for CellSimBench integration.
Optimal transport flow matching model operating in PCA space with gene embeddings.
"""

import logging
import json
import functools
import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import optax

import cellflow
from cellflow.model import CellFlow
from cellflow.utils import match_linear
from cellflow.preprocessing import centered_pca, project_pca, reconstruct_pca

from cellsimbench.utils.utils import PathEncoder
from cellsimbench.core.data_manager import DataManager
from utils import (
    prepare_gene_embeddings,
    filter_cells_by_embedding_availability,
    prepare_cellflow_obs,
)

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class CellFlowWrapper:
    """CellFlow model wrapper for CellSimBench integration."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_manager = DataManager(self.config)
        self.mode = self.config['mode']
        self.hyperparams = self.config['hyperparameters']

    # =========================================================================
    # Training
    # =========================================================================
    def train(self):
        """Train CellFlow model."""
        log.info("Starting CellFlow training process...")

        # Load data
        log.info("Loading CellSimBench data...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")

        # Check that embeddings exist
        embedding_key = self.hyperparams['embedding_key']
        if embedding_key not in adata.uns:
            raise ValueError(
                f"Embedding key '{embedding_key}' not found in adata.uns. "
                f"Available keys: {list(adata.uns.keys())}"
            )

        # ── Prepare CellFlow-format obs columns ──
        adata = self._convert_to_cellflow_format(adata, embedding_key)

        # ── Build train / val / test splits ──
        adata_train, adata_val, adata_test = self._build_splits(adata)

        # ── Compute PCA on training data, project val/test ──
        n_comps = self.hyperparams['n_pca_components']
        log.info(f"Computing centered PCA ({n_comps} components) on training data...")
        centered_pca(adata_train, method="rapids", keep_centered_data=False, n_comps=n_comps)
        project_pca(adata_val, ref_adata=adata_train)
        project_pca(adata_test, ref_adata=adata_train)

        # ── Initialize CellFlow ──
        log.info("Initializing CellFlow model...")
        cf = CellFlow(adata_train, solver="otfm")

        cf.prepare_data(
            sample_rep="X_pca",
            control_key="is_control",
            perturbation_covariates={
                "genetic_perturbation": ("gene_target_1", "gene_target_2")
            },
            perturbation_covariate_reps={
                "genetic_perturbation": "gene_embeddings"
            },
            max_combination_length=self.hyperparams['max_combination_length'],
            null_value=0.0,
        )

        # ── Prepare validation data ──
        cf.prepare_validation_data(
            adata_train,
            name="train",
            n_conditions_on_log_iteration=1,
            n_conditions_on_train_end=1,
        )
        cf.prepare_validation_data(
            adata_val,
            name="val",
            n_conditions_on_log_iteration=1,
            n_conditions_on_train_end=1,
        )

        # ── Prepare model architecture ──
        self._prepare_model(cf)

        # ── Setup callbacks (no W&B) ──
        metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
        decoded_metrics_callback = cellflow.training.PCADecodedMetrics(
            ref_adata=adata_train, metrics=["r_squared"]
        )
        callbacks = [metrics_callback, decoded_metrics_callback]

        # ── Train ──
        log.info(
            f"Starting training: {self.hyperparams['num_iterations']} iterations, "
            f"batch_size={self.hyperparams['batch_size']}"
        )
        cf.train(
            num_iterations=self.hyperparams['num_iterations'],
            batch_size=self.hyperparams['batch_size'],
            callbacks=callbacks,
            valid_freq=self.hyperparams['valid_freq'],
        )
        log.info("Training complete!")

        # ── Save model ──
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        cf.save(output_dir, file_prefix="cellflow", overwrite=True)
        log.info(f"Model saved to {output_dir}/cellflow_CellFlow.pkl")

        # ── Save PCA reference (needed for prediction) ──
        pca_ref_path = os.path.join(output_dir, "adata_train_pca_ref.h5ad")
        adata_train.write_h5ad(pca_ref_path)
        log.info(f"PCA reference saved to {pca_ref_path}")

        # ── Save metadata ──
        self._save_metadata(adata_train)

        log.info("CellFlow training completed successfully")

    # =========================================================================
    # Prediction
    # =========================================================================
    def predict(self):
        """Generate predictions using trained CellFlow model."""
        log.info("Starting CellFlow prediction process...")

        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # ── Load trained model ──
        model_pkl = model_path / "cellflow_CellFlow.pkl"
        log.info(f"Loading CellFlow model from {model_pkl}")
        cf = CellFlow.load(str(model_pkl))

        # ── Load PCA reference from training ──
        pca_ref_path = model_path / "adata_train_pca_ref.h5ad"
        log.info(f"Loading PCA reference from {pca_ref_path}")
        adata_train_ref = sc.read_h5ad(str(pca_ref_path))

        # ── Load full data ──
        log.info("Loading dataset for prediction...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")

        embedding_key = self.hyperparams['embedding_key']
        adata = self._convert_to_cellflow_format(adata, embedding_key)

        # ── Build test data with control cells ──
        test_conditions = self.config['test_conditions']
        n_ctrl_test = self.hyperparams['n_control_cells_test']

        # Get control cells
        ctrl_mask = adata.obs["is_control"].to_numpy()
        ctrl_idx = adata.obs.index[ctrl_mask]

        seed = self.hyperparams['seed']
        np.random.seed(seed)

        # Subsample controls if we have more than needed
        if len(ctrl_idx) > n_ctrl_test:
            ctrl_idx_sub = np.random.choice(ctrl_idx, size=n_ctrl_test, replace=False)
        else:
            ctrl_idx_sub = ctrl_idx

        # Get test perturbed cells (cells whose condition is in test_conditions)
        test_pert_mask = (
            adata.obs.condition.astype(str).isin(test_conditions)
            & ~ctrl_mask
        )
        test_pert_idx = adata.obs.index[test_pert_mask]

        adata_test = adata[list(ctrl_idx_sub) + list(test_pert_idx)].copy()
        log.info(
            f"Test set: {len(ctrl_idx_sub)} control + {len(test_pert_idx)} perturbed "
            f"= {adata_test.shape[0]} cells"
        )

        # ── Project test data into PCA space using training reference ──
        project_pca(adata_test, ref_adata=adata_train_ref)

        # ── Run prediction ──
        adata_ctrl = adata_test[adata_test.obs["is_control"].to_numpy()].copy()

        # Build covariate_data: unique test conditions (perturbed cells only)
        covariate_data = (
            adata_test[~adata_test.obs["is_control"].to_numpy()]
            .obs.drop_duplicates(subset=["condition"])
        )
        log.info(f"Predicting {len(covariate_data)} test conditions...")

        preds = cf.predict(
            adata=adata_ctrl,
            sample_rep="X_pca",
            condition_id_key="condition",
            covariate_data=covariate_data,
        )

        # ── Build AnnData from predictions ──
        adata_pred_list = []
        for cond, array in preds.items():
            obs_data = pd.DataFrame({"condition": [cond] * array.shape[0]})
            adata_pred = ad.AnnData(
                X=np.empty((len(array), adata_train_ref.n_vars)),
                obs=obs_data,
            )
            adata_pred.obsm["X_pca"] = np.squeeze(array)
            adata_pred_list.append(adata_pred)

        adata_preds = ad.concat(adata_pred_list)
        adata_preds.var_names = adata_train_ref.var_names

        # ── Reconstruct to gene space ──
        log.info("Reconstructing predictions from PCA to gene space...")
        reconstruct_pca(adata_preds, use_rep="X_pca", ref_adata=adata_train_ref)
        adata_preds.X = adata_preds.layers["X_recon"]

        log.info(f"Generated {adata_preds.shape[0]} predicted cells across {len(preds)} conditions")

        # ── Convert to CellSimBench format (mean per condition) ──
        predictions_adata = self._convert_to_cellsimbench_format(adata_preds, adata)

        # ── Save predictions ──
        output_path = self.config['output_path']
        log.info(f"Saving predictions to {output_path}")
        predictions_adata.write_h5ad(output_path)

        log.info("Prediction completed successfully")

    # =========================================================================
    # Internal helpers
    # =========================================================================
    def _convert_to_cellflow_format(self, adata: sc.AnnData, embedding_key: str) -> sc.AnnData:
        """Convert CellSimBench data to CellFlow format.
        
        Adds obs columns (is_control, gene_target_1, gene_target_2), prepares
        gene embeddings as dict, and filters cells with missing embeddings.
        """
        # Remove ctrl_synthetic_mean cells if present (not real data)
        if "ctrl_synthetic_mean" in adata.obs.condition.astype(str).values:
            n_before = len(adata)
            adata = adata[adata.obs.condition.astype(str) != "ctrl_synthetic_mean"].copy()
            log.info(f"Removed ctrl_synthetic_mean cells: {n_before} -> {len(adata)}")

        # Add CellFlow-required obs columns
        prepare_cellflow_obs(adata)

        # Prepare gene embeddings (DataFrame -> dict)
        gene_embeddings = prepare_gene_embeddings(adata, embedding_key)
        adata.uns["gene_embeddings"] = gene_embeddings

        # Filter cells missing embeddings
        adata = filter_cells_by_embedding_availability(adata, gene_embeddings)

        log.info(f"Final data shape: {adata.shape}")
        return adata

    def _build_splits(self, adata: sc.AnnData):
        """Build train/val/test AnnData splits with subsampled controls.
        
        Returns:
            Tuple of (adata_train, adata_val, adata_test).
        """
        split_name = self.config['split_name']
        train_conditions = set(self.config['train_conditions'])
        val_conditions = set(self.config['val_conditions'])
        test_conditions = set(self.config['test_conditions'])

        n_ctrl_train = self.hyperparams['n_control_cells_train']
        n_ctrl_val = self.hyperparams['n_control_cells_val']
        n_ctrl_test = self.hyperparams['n_control_cells_test']
        seed = self.hyperparams['seed']

        is_ctrl = adata.obs["is_control"].to_numpy()
        condition_str = adata.obs.condition.astype(str)
        split_col = adata.obs[split_name].astype(str)

        # Get control cell indices and subsample
        ctrl_idx = adata.obs.index[is_ctrl]
        np.random.seed(seed)

        def _subsample_ctrl(n):
            if len(ctrl_idx) <= n:
                return list(ctrl_idx)
            return list(np.random.choice(ctrl_idx, size=n, replace=False))

        ctrl_train = _subsample_ctrl(n_ctrl_train)
        ctrl_val = _subsample_ctrl(n_ctrl_val)
        ctrl_test = _subsample_ctrl(n_ctrl_test)

        # Train: perturbed cells from train split
        train_pert_mask = (
            (split_col == "train") & ~is_ctrl & condition_str.isin(train_conditions)
        )
        train_pert_idx = list(adata.obs.index[train_pert_mask])
        adata_train = adata[ctrl_train + train_pert_idx].copy()

        # Val: perturbed cells from val split
        val_pert_mask = (
            (split_col == "val") & ~is_ctrl & condition_str.isin(val_conditions)
        )
        val_pert_idx = list(adata.obs.index[val_pert_mask])
        adata_val = adata[ctrl_val + val_pert_idx].copy()

        # Test: perturbed cells from test split
        test_pert_mask = (
            (split_col == "test") & ~is_ctrl & condition_str.isin(test_conditions)
        )
        test_pert_idx = list(adata.obs.index[test_pert_mask])
        adata_test = adata[ctrl_test + test_pert_idx].copy()

        log.info(f"Train set: {adata_train.shape[0]} cells ({len(ctrl_train)} ctrl + {len(train_pert_idx)} pert)")
        log.info(f"Val set:   {adata_val.shape[0]} cells ({len(ctrl_val)} ctrl + {len(val_pert_idx)} pert)")
        log.info(f"Test set:  {adata_test.shape[0]} cells ({len(ctrl_test)} ctrl + {len(test_pert_idx)} pert)")

        return adata_train, adata_val, adata_test

    def _prepare_model(self, cf: CellFlow):
        """Configure CellFlow model architecture from hyperparameters."""
        hp = self.hyperparams

        # Perturbation encoder MLP (before pooling)
        layers_before_pool = {
            "genetic_perturbation": {
                "layer_type": "mlp",
                "dims": hp['layers_before_pool_dims'],
                "dropout_rate": hp['layers_before_pool_dropout'],
            },
        }

        # After pooling MLP
        layers_after_pool = {
            "layer_type": "mlp",
            "dims": hp['layers_after_pool_dims'],
            "dropout_rate": hp['layers_after_pool_dropout'],
        }

        # OT matching function
        match_fn = functools.partial(
            match_linear,
            epsilon=hp['epsilon'],
            tau_a=hp['tau_a'],
            tau_b=hp['tau_b'],
        )

        # Optimizer: Adam with gradient accumulation
        optimizer = optax.MultiSteps(
            optax.adam(hp['learning_rate']),
            hp['multi_steps'],
        )

        cf.prepare_model(
            # Condition encoder
            condition_mode="deterministic",
            regularization=0.0,
            condition_embedding_dim=hp['condition_embedding_dim'],
            cond_output_dropout=hp['cond_output_dropout'],
            # Pooling
            pooling=hp['pooling'],
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            # Time encoder
            time_freqs=hp['time_freqs'],
            time_encoder_dims=hp['time_encoder_dims'],
            time_encoder_dropout=0.0,
            # Flow network
            hidden_dims=hp['hidden_dims'],
            hidden_dropout=0.0,
            decoder_dims=hp['decoder_dims'],
            decoder_dropout=hp['decoder_dropout'],
            # Conditioning and flow
            conditioning="concatenation",
            probability_path={"constant_noise": 1.0},
            match_fn=match_fn,
            linear_projection_before_concatenation=False,
            layer_norm_before_concatenation=False,
            # Optimizer
            optimizer=optimizer,
        )
        log.info("CellFlow model architecture prepared")

    def _convert_to_cellsimbench_format(
        self, adata_preds: ad.AnnData, original_adata: sc.AnnData
    ) -> sc.AnnData:
        """Convert CellFlow predictions to CellSimBench format (mean per condition)."""
        test_conditions = self.config['test_conditions']

        prediction_list = []
        condition_list = []

        for condition in test_conditions:
            if condition == "control":
                continue
            mask = adata_preds.obs["condition"].astype(str) == condition
            if mask.sum() == 0:
                log.warning(f"No predictions found for condition '{condition}', skipping")
                continue

            pred_cells = adata_preds[mask].X
            if hasattr(pred_cells, "toarray"):
                pred_cells = pred_cells.toarray()

            # Take mean across predicted cells for this condition
            pred_mean = np.mean(pred_cells, axis=0)
            prediction_list.append(pred_mean)
            condition_list.append(condition)

        if not prediction_list:
            raise ValueError("No valid predictions found for test conditions")

        prediction_matrix = np.vstack(prediction_list)
        obs_df = pd.DataFrame({"condition": condition_list})
        adata_pred = sc.AnnData(X=prediction_matrix, obs=obs_df)
        adata_pred.var_names = original_adata.var_names

        log.info(f"Converted predictions: {adata_pred.shape[0]} conditions, {adata_pred.shape[1]} genes")
        return adata_pred

    def _save_metadata(self, adata_train: sc.AnnData):
        """Save training metadata."""
        metadata = {
            "model_type": "CellFlow (Optimal Transport Flow Matching)",
            "embedding_key": self.hyperparams['embedding_key'],
            "config": self.config,
            "data_shape": list(adata_train.shape),
            "n_pca_components": self.hyperparams['n_pca_components'],
            "num_iterations": self.hyperparams['num_iterations'],
        }

        output_dir = Path(self.config['output_dir'])
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder)
        log.info("Metadata saved")
