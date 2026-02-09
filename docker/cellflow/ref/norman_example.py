from typing import Any
import warnings
from pandas.errors import SettingWithCopyWarning
import matplotlib
import os

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import functools
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import cellflow
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance
import optax

# =============================================================================
# Load and prepare data
# =============================================================================
adata = sc.read_h5ad("data/norman19_processed_complete.h5ad")

# Remove ctrl_synthetic_mean cells (not real data)
adata = adata[adata.obs.condition != "ctrl_synthetic_mean"].copy()

# Create is_control boolean column
adata.obs["is_control"] = adata.obs.condition == "control"

# Split condition column into gene_target_1 and gene_target_2
# For combos like "AHR+FEV" -> gene_target_1="AHR", gene_target_2="FEV"
# For singles like "AHR" -> gene_target_1="AHR", gene_target_2="control"
# For control -> gene_target_1="control", gene_target_2="control"
def split_condition(cond):
    if cond == "control":
        return "control", "control"
    if "+" in cond:
        parts = cond.split("+")
        return parts[0], parts[1]
    return cond, "control"

# Convert categorical to string to avoid pandas MultiIndex issues with apply
condition_str = adata.obs.condition.astype(str)
splits = [split_condition(c) for c in condition_str]
adata.obs["gene_target_1"] = [s[0] for s in splits]
adata.obs["gene_target_2"] = [s[1] for s in splits]

# Convert ESM2 embeddings from DataFrame to dict format
esm2_df = adata.uns['embeddings_esm2_mean']
gene_embeddings = {gene: np.array(esm2_df.loc[gene]) for gene in esm2_df.index}
# Add control token (zeros with same dimension)
gene_embeddings["control"] = np.zeros(esm2_df.shape[1])
adata.uns["gene_embeddings"] = gene_embeddings

# Get all unique perturbation genes (excluding "control")
all_pert_genes = set(adata.obs.gene_target_1.unique()) | set(adata.obs.gene_target_2.unique())
all_pert_genes.discard("control")

# Find which perturbation genes have embeddings
available_genes = set(gene_embeddings.keys())
missing_genes = all_pert_genes - available_genes
genes_with_embeddings = all_pert_genes & available_genes

print(f"Total unique perturbation genes: {len(all_pert_genes)}")
print(f"Genes with embeddings: {len(genes_with_embeddings)}")
print(f"Genes missing embeddings: {len(missing_genes)}")
if missing_genes:
    print(f"Missing genes: {sorted(missing_genes)[:20]}...")  # Show first 20

# Filter to only keep cells where both gene targets have embeddings (or are "control")
valid_genes = genes_with_embeddings | {"control"}
mask = (
    adata.obs.gene_target_1.isin(valid_genes) & 
    adata.obs.gene_target_2.isin(valid_genes)
)
n_before = len(adata)
adata = adata[mask].copy()
n_after = len(adata)
print(f"\nFiltered from {n_before} to {n_after} cells ({n_before - n_after} cells removed)")

print(f"\nFinal data shape: {adata.shape}")
print(f"Number of control cells: {adata.obs.is_control.sum()}")
print(f"Number of perturbed cells: {(~adata.obs.is_control).sum()}")
print(f"Number of unique conditions: {adata.obs.condition.nunique()}")
print(f"Gene embedding dimension: {esm2_df.shape[1]}")

# =============================================================================
# Setup train/val/test splits
# =============================================================================
# Using split_fold_0 which ensures all single genes in train, some combos in train,
# some in val, and half in test

adata_train = adata[adata.obs.split_fold_0 == "train"].copy()

# For validation/test, we need control cells to generate predictions from
# BUT we subsample control cells to speed up validation (they're redundant)
n_control_cells_for_val = 300
n_control_cells_for_train = 2048
n_control_cells_for_test = 300

# Get control cell indices and subsample
ctrl_idx = adata.obs[adata.obs.is_control].index
np.random.seed(42)
ctrl_idx_subsampled_train = np.random.choice(ctrl_idx, size=n_control_cells_for_train, replace=False)
ctrl_idx_subsampled_val = np.random.choice(ctrl_idx, size=n_control_cells_for_val, replace=False)
ctrl_idx_subsampled_test = np.random.choice(ctrl_idx, size=n_control_cells_for_test, replace=False)

# Train set: train perturbed cells
train_pert_idx = adata.obs[(adata.obs.split_fold_0 == "train") & (~adata.obs.is_control)].index
adata_train = adata[ list(ctrl_idx_subsampled_train) + list(train_pert_idx)].copy()

# Val set: subsampled controls + val perturbed cells
val_pert_idx = adata.obs[(adata.obs.split_fold_0 == "val") & (~adata.obs.is_control)].index
adata_val = adata[list[Any](ctrl_idx_subsampled_val) + list(val_pert_idx)].copy()

# Test set: subsampled controls + test perturbed cells  
test_pert_idx = adata.obs[(adata.obs.split_fold_0 == "test") & (~adata.obs.is_control)].index
adata_test = adata[list(ctrl_idx_subsampled_test) + list(test_pert_idx)].copy()

print(f"\nTrain set: {adata_train.shape[0]} cells")
print(f"Val set: {adata_val.shape[0]} cells ({n_control_cells_for_val} control + {len(val_pert_idx)} perturbed)")
print(f"Test set: {adata_test.shape[0]} cells ({n_control_cells_for_val} control + {len(test_pert_idx)} perturbed)")

# =============================================================================
# Compute PCA on training data
# =============================================================================
# Use centered PCA based on training data, then project val/test
centered_pca(adata_train, method="rapids", keep_centered_data=False, n_comps=50)
project_pca(adata_val, ref_adata=adata_train)
project_pca(adata_test, ref_adata=adata_train)

# =============================================================================
# Setup CellFlow model
# =============================================================================
cf = CellFlow(adata_train, solver="otfm")

cf.prepare_data(
    sample_rep="X_pca",
    control_key="is_control",
    perturbation_covariates={"genetic_perturbation": ("gene_target_1", "gene_target_2")},
    perturbation_covariate_reps={"genetic_perturbation": "gene_embeddings"},
    max_combination_length=2,
    null_value=0.0,
)

# Prepare validation data
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

# =============================================================================
# Prepare model architecture (from CellFlow paper - Norman19 task)
# =============================================================================
# Perturbation encoder: project ESM2 embeddings (5120-dim) through MLP
layers_before_pool = {
    "genetic_perturbation": {
        "layer_type": "mlp",
        "dims": [1024, 1024],
        "dropout_rate": 0.5,
    },
}

# After pooling MLP
layers_after_pool = {
    "layer_type": "mlp",
    "dims": [1024, 1024],
    "dropout_rate": 0.2,
}

# OT matching function
match_fn = functools.partial(match_linear, epsilon=0.1, tau_a=1.0, tau_b=1.0)

cf.prepare_model(
    # Condition encoder
    condition_mode="deterministic",
    regularization=0.0,
    condition_embedding_dim=1024,
    cond_output_dropout=0.9,
    
    # Pooling (attention token for learned aggregation of combos)
    pooling="attention_token",
    layers_before_pool=layers_before_pool,
    layers_after_pool=layers_after_pool,
    
    # Time encoder
    time_freqs=1024,
    time_encoder_dims=[2048, 2048, 2048],
    time_encoder_dropout=0.0,
    
    # Flow network (large!)
    hidden_dims=[4096, 4096, 4096],
    hidden_dropout=0.0,
    decoder_dims=[4096, 4096, 4096],
    decoder_dropout=0.2,
    
    # Conditioning and flow
    conditioning="concatenation",
    probability_path={"constant_noise": 1.0},
    match_fn=match_fn,
    linear_projection_before_concatenation=False,  # Paper default
    layer_norm_before_concatenation=False,         # Paper default
    
    # Optimizer: lr=5e-5 with 20 gradient accumulation steps
    optimizer=optax.MultiSteps(optax.adam(5e-5), 20),
)

# =============================================================================
# Setup callbacks for metrics
# =============================================================================
metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
decoded_metrics_callback = cellflow.training.PCADecodedMetrics(
    ref_adata=adata_train, metrics=["r_squared"]
)

# Wandb logging for real-time monitoring
wandb_callback = cellflow.training.WandbLogger(
    project="cellflow",
    out_dir="./wandb_logs",
    config={
        "dataset": "norman19",
        "num_conditions": adata_train.obs.condition.nunique(),
        "num_cells_train": len(adata_train),
        "embedding_type": "esm2_mean",
        "embedding_dim": 5120,
        # Paper hyperparameters
        "condition_embedding_dim": 1024,
        "hidden_dims": [4096, 4096, 4096],
        "decoder_dims": [4096, 4096, 4096],
        "pooling": "attention_token",
        "layers_before_pool_dims": [1024, 1024],
        "layers_before_pool_dropout": 0.5,
        "layers_after_pool_dims": [1024, 1024],
        "layers_after_pool_dropout": 0.2,
        "cond_output_dropout": 0.9,
        "decoder_dropout": 0.2,
        "flow_noise": 1.0,
        "epsilon": 0.1,
        "learning_rate": 5e-5,
        "multi_steps": 20,
        "num_iterations": 200_000,
        "batch_size": 1024,
    },
    entity="henrymiller2024-none",
    name="norman19_paper_hparams",
)

callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]

# =============================================================================
# Train the model
# =============================================================================
print("\nStarting training...")
cf.train(
    num_iterations=200_000,  # Paper: 200k gradient updates
    batch_size=1024,
    callbacks=callbacks,
    valid_freq=50_000,  # Validate every 50k steps
)

print("\nTraining complete!")
print(f"Training logs available: {list(cf.trainer.training_logs.keys())}")

# =============================================================================
# Save the model
# =============================================================================
import os
os.makedirs("./models", exist_ok=True)
cf.save("./models", file_prefix="norman19", overwrite=True)
print("Model saved to ./models/norman19_CellFlow.pkl")

# =============================================================================
# Load model (if needed later)
# =============================================================================
# cf = CellFlow.load("./models/norman19_CellFlow.pkl")

# =============================================================================
# Inference on test set
# =============================================================================
# Get control cells for prediction (use full control set for final predictions)
adata_ctrl_for_prediction = adata_test[adata_test.obs["is_control"].to_numpy()].copy()

# Get unique test conditions (perturbed cells only, deduplicated)
covariate_data = adata_test[~adata_test.obs["is_control"].to_numpy()].obs.drop_duplicates(subset=["condition"])
print(f"\nPredicting {len(covariate_data)} test conditions...")

# Make predictions - returns dict of {condition: predicted_cells}
preds = cf.predict(
    adata=adata_ctrl_for_prediction, 
    sample_rep="X_pca", 
    condition_id_key="condition", 
    covariate_data=covariate_data
)

# =============================================================================
# Build AnnData from predictions
# =============================================================================
adata_preds = []
for cond, array in preds.items():
    obs_data = pd.DataFrame({
        'condition': [cond] * array.shape[0]
    })
    adata_pred = ad.AnnData(X=np.empty((len(array), adata_train.n_vars)), obs=obs_data)
    adata_pred.obsm["X_pca"] = np.squeeze(array)
    adata_preds.append(adata_pred)

adata_preds = ad.concat(adata_preds)
adata_preds.var_names = adata_train.var_names

# Reconstruct to gene space
reconstruct_pca(adata_preds, use_rep="X_pca", ref_adata=adata_train)
adata_preds.X = adata_preds.layers["X_recon"]

print(f"Generated {adata_preds.shape[0]} predicted cells")

# =============================================================================
# Compute metrics on test set
# =============================================================================
# Get ground truth test data (full, not subsampled) - use ORIGINAL gene expression
adata_test_full = adata[(adata.obs.split_fold_0 == "test") | (adata.obs.is_control)].copy()
# Project to PCA space for energy distance computation
project_pca(adata_test_full, ref_adata=adata_train)

# Compute metrics per condition
# R² computed in gene space: predictions vs ORIGINAL ground truth (not PCA-reconstructed)
# This measures full end-to-end performance including PCA reconstruction error

results = []
for cond in adata_preds.obs["condition"].unique():
    # Predicted cells (already in gene space after reconstruct_pca)
    pred_cells = adata_preds[adata_preds.obs["condition"] == cond].X
    
    # Ground truth: ORIGINAL gene expression (not put through PCA encode/decode)
    true_cells = adata_test_full[adata_test_full.obs["condition"] == cond].X
    
    if hasattr(true_cells, 'toarray'):
        true_cells = true_cells.toarray()
    if hasattr(pred_cells, 'toarray'):
        pred_cells = pred_cells.toarray()
    
    if len(true_cells) > 0:
        # R² in gene expression space (predictions vs original ground truth)
        r2 = compute_r_squared(true_cells, pred_cells)
        
        # Energy distance in PCA space (latent space metric)
        e_dist = compute_e_distance(
            adata_test_full[adata_test_full.obs["condition"] == cond].obsm["X_pca"],
            adata_preds[adata_preds.obs["condition"] == cond].obsm["X_pca"]
        )
        results.append({"condition": cond, "r_squared": r2, "e_distance": e_dist})

df_results = pd.DataFrame(results)
print("\n=== Test Set Metrics ===")
print(df_results)
print(f"\nMean R²: {df_results['r_squared'].mean():.4f}")
print(f"Mean E-distance: {df_results['e_distance'].mean():.4f}")
