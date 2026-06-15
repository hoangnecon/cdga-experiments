"""
Component: Seed Utilities
Location: shared/utils/seed_utils.py

Ref: rules/CONVENTIONS.md Section 7
"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Must be called BEFORE:
        - Model initialization (weight init uses random)
        - DataLoader creation (shuffle order uses random)
        - Any augmentation setup

    Args:
        seed: Integer seed. Must be stored in config['project']['seed'].

    Note:
        torch.backends.cudnn.deterministic = True may slightly reduce
        performance but is required for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Must be False when deterministic=True
