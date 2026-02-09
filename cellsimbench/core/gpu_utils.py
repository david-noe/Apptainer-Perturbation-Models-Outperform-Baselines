"""
GPU management utilities for parallel fold processing.

Provides GPU detection, assignment, and environment management for
parallel training and inference across multiple folds.
"""

import os
import logging
from typing import List, Dict, Tuple

log = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get available GPU indices respecting CUDA_VISIBLE_DEVICES."""
    import pynvml
    
    # Check if CUDA_VISIBLE_DEVICES is set
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if cuda_visible is not None:
        # Parse CUDA_VISIBLE_DEVICES
        if cuda_visible.strip() == "":
            # Empty string means no GPUs
            log.info("CUDA_VISIBLE_DEVICES is empty - no GPUs available")
            return []
        
        try:
            # Parse comma-separated GPU indices
            visible_gpus = [int(gpu.strip()) for gpu in cuda_visible.split(',')]
            log.info(f"CUDA_VISIBLE_DEVICES={cuda_visible} - available GPUs: {visible_gpus}")
            return visible_gpus
        except ValueError as e:
            log.error(f"Invalid CUDA_VISIBLE_DEVICES format '{cuda_visible}': {e}")
            return []
    
    else:
        # CUDA_VISIBLE_DEVICES not set - detect all GPUs using pynvml
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            all_gpus = list(range(gpu_count))
            log.info(f"CUDA_VISIBLE_DEVICES not set - detected {gpu_count} GPUs: {all_gpus}")
            return all_gpus
        except Exception as e:
            log.warning(f"Failed to detect GPUs with pynvml: {e}")
            return []


def calculate_gpu_assignment(fold_indices: List[int], available_gpus: List[int]) -> Dict[int, int]:
    """Calculate round-robin GPU assignment for folds.
    
    Args:
        fold_indices: List of fold indices to train
        available_gpus: List of available GPU IDs (respecting CUDA_VISIBLE_DEVICES)
        
    Returns:
        Dict mapping fold_idx -> gpu_id
    """
    if not available_gpus:
        raise ValueError("No GPUs available for training")
    
    gpu_assignment = {}
    for i, fold_idx in enumerate(fold_indices):
        gpu_id = available_gpus[i % len(available_gpus)]
        gpu_assignment[fold_idx] = gpu_id
    
    log.info(f"GPU assignment (round-robin): {gpu_assignment}")
    return gpu_assignment


def calculate_exclusive_gpu_assignment(
    fold_indices: List[int], available_gpus: List[int]
) -> Tuple[Dict[int, int], int]:
    """Assign GPUs round-robin but cap concurrency to len(available_gpus).

    This ensures at most one job runs per GPU at a time. When there are more
    folds than GPUs, excess jobs wait in the ThreadPoolExecutor queue until a
    GPU becomes free.

    Args:
        fold_indices: List of fold indices to process.
        available_gpus: List of available GPU IDs.

    Returns:
        Tuple of (gpu_assignment dict mapping fold_idx -> gpu_id,
                  max_concurrent_jobs).
    """
    gpu_assignment = calculate_gpu_assignment(fold_indices, available_gpus)
    max_concurrent = len(available_gpus)
    log.info(
        f"Exclusive GPU scheduling: {len(fold_indices)} folds, "
        f"{max_concurrent} GPUs -> max {max_concurrent} concurrent jobs"
    )
    return gpu_assignment, max_concurrent


# Note: gpu_environment() context manager removed since we now use Docker device_requests
# for proper GPU assignment instead of thread-unsafe environment variables
