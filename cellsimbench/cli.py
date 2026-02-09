"""
Command-line interface for CellSimBench.

Provides the main entry points for training models and running benchmarks
through Hydra-based configuration management.
"""

import logging
import sys
from typing import Optional
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger(__name__)

# Get the absolute path to the configs directory
CONFIGS_PATH = str(Path(__file__).parent / "configs")


def main() -> None:
    """Main CLI entrypoint for CellSimBench with subcommands.
    
    Provides 'train' and 'benchmark' subcommands for model training
    and evaluation respectively.
    """
    
    if len(sys.argv) < 2:
        print("Usage: cellsimbench {train|benchmark} [options]")
        print("  train     - Train models independently")
        print("  benchmark - Benchmark pre-trained models")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        # Remove 'train' from argv and call train main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
    elif command == "benchmark":
        # Remove 'benchmark' from argv and call benchmark main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        benchmark_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, benchmark")
        sys.exit(1)


@hydra.main(version_base=None, config_path=CONFIGS_PATH, config_name="config")
def train_main(cfg: DictConfig) -> None:
    """Training entrypoint for CellSimBench.
    
    Trains a single model based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration for training including model, dataset,
             and training parameters.
             
    Raises:
        Exception: If training fails for any reason.
    """
    
    try:
        from cellsimbench.core.training_runner import TrainingRunner
        
        # Add default training configuration if not present
        if not hasattr(cfg, 'training'):
            from omegaconf import OmegaConf, open_dict
            with open_dict(cfg):
                cfg.training = {
                    'output_dir': f'models/{cfg.model.name}_{cfg.dataset.name}/',
                    'save_intermediate': True
                }
        
        # Update experiment name for training
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.experiment.name = f"train_{cfg.model.name}_{cfg.dataset.name}"
            cfg.experiment.description = f"Train {cfg.model.name} on {cfg.dataset.name} dataset"
        
        # Create and run training
        runner = TrainingRunner(cfg)
        model_path = runner.train_model()
        
        log.info(f"Training completed. Model saved to: {model_path}")
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@hydra.main(version_base=None, config_path=CONFIGS_PATH, config_name="config")
def benchmark_main(cfg: DictConfig) -> float:
    """Benchmarking entrypoint for CellSimBench.
    
    Runs benchmarks on pre-trained models and generates evaluation metrics.
    
    Args:
        cfg: Hydra configuration for benchmarking including model(s),
             dataset, and output settings.
        
    Returns:
        Primary metric value (0.0 on success, -inf on failure) for
        Hydra optimization compatibility.
    """
    
    try:
        from cellsimbench.core.benchmark import BenchmarkRunner
        from cellsimbench.utils.hash_utils import get_model_path_for_config
        from omegaconf import OmegaConf
        
        # Handle modelgroup configs that reference multiple models
        if hasattr(cfg, 'modelgroup'):
            # Load modelgroup config and expand to individual model configs
            from hydra import compose
            
            # Create models list from modelgroup config
            model_configs = []
            for model_name in cfg.modelgroup.models:
                # Load the individual model config
                try:
                    model_cfg = compose(config_name="config", overrides=[f"model={model_name}"])
                    model_configs.append(model_cfg.model)
                except Exception as e:
                    log.error(f"Failed to load model config '{model_name}': {e}")
                    raise
            
            from omegaconf import open_dict
            with open_dict(cfg):
                cfg.models = model_configs
                # Remove modelgroup and model configs to avoid conflicts
                # Use delattr to fully remove from config
                if 'modelgroup' in cfg:
                    delattr(cfg, 'modelgroup')
                if 'model' in cfg:
                    delattr(cfg, 'model')
        
        # Handle single model config - just ensure training config exists
        elif hasattr(cfg, 'model') and not hasattr(cfg, 'models'):
            if cfg.model.type == 'docker':
                # Add default training config if not present for hash calculation
                if not hasattr(cfg, 'training'):
                    from omegaconf import open_dict
                    with open_dict(cfg):
                        cfg.training = {
                            'output_dir': f'models/{cfg.model.name}_{cfg.dataset.name}/',
                            'save_intermediate': True
                        }
        
        # Handle models list (from modelgroup)
        if hasattr(cfg, 'models'):
            # Calculate model_path for each model that doesn't have one
            for i, model_config in enumerate(cfg.models):
                if model_config.type == 'docker' and 'model_path' not in model_config:
                    # Create individual training config for each model (same as single model approach)
                    individual_training_config = OmegaConf.create({
                        'output_dir': f'models/{model_config.name}_{cfg.dataset.name}/',
                        'save_intermediate': True
                    })
                    
                    # Calculate the model path using the full model config and individual training config
                    model_path = str(get_model_path_for_config(cfg.dataset, model_config, individual_training_config))
                    
                    # Add the calculated path to the config
                    from omegaconf import open_dict
                    with open_dict(cfg):
                        cfg.models[i]['model_path'] = model_path
            
            # Add default training config if not present (for other parts of the code)
            if not hasattr(cfg, 'training'):
                from omegaconf import open_dict
                with open_dict(cfg):
                    cfg.training = {
                        'output_dir': f'models/{cfg.dataset.name}/',
                        'save_intermediate': True
                    }
            
            # Update experiment name for multi-model benchmarking
            from omegaconf import open_dict
            with open_dict(cfg):
                model_names = [getattr(m, 'display_name', m.name) for m in cfg.models]
                cfg.experiment.name = f"benchmark_{'_'.join(model_names)}_{cfg.dataset.name}"
                cfg.experiment.description = f"Benchmark {', '.join(model_names)} on {cfg.dataset.name} dataset"
        
        # Check if we should just calculate and print hashes
        calculate_hash_only = getattr(cfg, 'calculate_hash_only', False)
        if calculate_hash_only:
            _print_hashes(cfg)
            return 0.0
        
        # Create and run benchmark
        runner = BenchmarkRunner(cfg)
        results = runner.run_benchmark()
        
        log.info(f"Benchmark completed.")
        
        return 0.0
        
    except Exception as e:
        log.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return float('-inf')  # Return very low score on failure for optimization


def _print_hashes(cfg: DictConfig) -> None:
    """Calculate and print all training/inference hashes for each fold, then stop.
    
    This is a diagnostic tool to inspect what cache keys the system would
    compute without actually running any training or inference.
    
    Args:
        cfg: Hydra configuration (same as benchmark_main receives).
    """
    from omegaconf import OmegaConf, open_dict
    from cellsimbench.utils.hash_utils import calculate_input_hash, get_model_path_for_config, calculate_inference_hash
    from cellsimbench.core.data_manager import DataManager
    import json
    
    print("\n" + "=" * 80)
    print("HASH DIAGNOSTIC MODE (+calculate_hash_only=true)")
    print("=" * 80)
    
    # Determine folds
    if hasattr(cfg.dataset, 'folds'):
        fold_indices = list(range(len(cfg.dataset.folds)))
    else:
        raise ValueError("No folds defined in dataset config")
    
    # Build list of (model_config, training_config) pairs to check
    model_entries = []
    if hasattr(cfg, 'models'):
        for model_config in cfg.models:
            if model_config.type == 'baselines_only':
                continue
            clean = OmegaConf.to_object(model_config)
            if 'model_path' in clean:
                del clean['model_path']
            clean = OmegaConf.create(clean)
            training_config = OmegaConf.create({
                'output_dir': f'models/{model_config.name}_{cfg.dataset.name}/',
                'save_intermediate': True
            })
            model_entries.append((model_config.name, clean, training_config))
    elif hasattr(cfg, 'model'):
        if cfg.model.type != 'baselines_only':
            training_config = cfg.training
            model_entries.append((cfg.model.name, cfg.model, training_config))
    
    if not model_entries:
        print("No docker models found in config.")
        return
    
    # Load data manager for perturbation conditions (needed for inference hash)
    dataset_config_dict = OmegaConf.to_object(cfg.dataset)
    data_manager = DataManager(dataset_config_dict)
    data_manager.load_dataset()
    
    data_file_path = Path(data_manager.config['data_path']).resolve()
    data_mtime = data_file_path.stat().st_mtime
    
    print(f"\nData file: {data_file_path}")
    print(f"Data mtime (st_mtime): {data_mtime}")
    
    for model_name, model_config, training_config in model_entries:
        print(f"\n{'─' * 80}")
        print(f"MODEL: {model_name}")
        print(f"{'─' * 80}")
        
        for fold_idx in fold_indices:
            fold_config = cfg.dataset.folds[fold_idx]
            fold_split = fold_config.split
            
            # Create fold-specific dataset config
            fold_dataset_config = OmegaConf.create(OmegaConf.to_object(cfg.dataset))
            with open_dict(fold_dataset_config):
                fold_dataset_config.split = fold_split
            
            # --- Training hash ---
            input_hash = calculate_input_hash(fold_dataset_config, model_config, training_config)
            model_path = get_model_path_for_config(fold_dataset_config, model_config, training_config)
            
            print(f"\n  Fold {fold_idx} ({fold_split}):")
            print(f"    Training hash (full):  {input_hash}")
            print(f"    Training hash (short): {input_hash[:12]}")
            print(f"    Model path:            {model_path.resolve()}")
            print(f"    Model path exists:     {model_path.resolve().exists()}")
            
            # --- Inference hash ---
            training_checkpoint_path = model_path / 'training_checkpoint.json'
            if not training_checkpoint_path.exists():
                print(f"    Inference hash:        CANNOT COMPUTE (no training_checkpoint.json at {training_checkpoint_path.resolve()})")
                continue
            
            # Read checkpoint to show its stored hash
            with open(training_checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            print(f"    Checkpoint input_hash: {checkpoint['input_hash']}")
            print(f"    Checkpoint timestamp:  {checkpoint['timestamp']}")
            
            hash_match = (checkpoint['input_hash'] == input_hash)
            print(f"    Training hash match:   {'YES' if hash_match else 'NO (MISMATCH!)'}")
            
            # Build pred_config the same way model_runner.py does
            conditions = data_manager.get_perturbation_conditions(fold_split)
            test_conditions = conditions['test']
            
            pred_config = {
                'mode': 'predict',
                'data_path': str(data_file_path),
                'model_path': str(model_path),
                'split_name': fold_split,
                'test_conditions': sorted(test_conditions),
                'covariate_key': data_manager.config['covariate_key'],
                'hyperparameters': OmegaConf.to_object(model_config.hyperparameters),
            }
            
            inference_hash = calculate_inference_hash(pred_config, training_checkpoint_path)
            cache_dir = model_path / 'predictions_cache' / f'inf_{inference_hash[:12]}'
            cached_predictions = cache_dir / 'predictions.h5ad'
            
            print(f"    Inference hash (full): {inference_hash}")
            print(f"    Inference hash (short):{inference_hash[:12]}")
            print(f"    Cache path:            {cache_dir.resolve()}")
            print(f"    Cache hit:             {cached_predictions.exists()}")
    
    print(f"\n{'=' * 80}")
    print("HASH DIAGNOSTIC COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main() 