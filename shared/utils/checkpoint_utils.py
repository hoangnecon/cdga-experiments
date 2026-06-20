"""
Component: Checkpoint Utilities
Location: shared/utils/checkpoint_utils.py

Ref: rules/STRUCTURE.md (Checkpoint section)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: dict,
    config: dict,
    metrics_history: list[dict],
    best_miou: float = 0.0,
    best_bf1: float = 0.0,
) -> None:
    """Save a checkpoint to disk.

    Saved structure (always consistent):
        {
            'epoch': int,
            'model_state_dict': OrderedDict,
            'optimizer_state_dict': dict,
            'scheduler_state_dict': dict,
            'metrics': dict,          # Metrics at this epoch
            'config': dict,           # Full config (frozen copy)
            'metrics_history': list,  # All epochs so far
        }

    Args:
        path: Full path to checkpoint file (e.g., run_dir/checkpoints/best_miou.pth).
        epoch: Current epoch (1-indexed).
        model: The model (or model wrapper — only backbone state_dict is saved).
        optimizer: Optimizer.
        scheduler: LR scheduler.
        metrics: Metric dict for this epoch (miou, mf1, oa, bf1_3, ...).
        config: Full config dict (will be stored as reference).
        metrics_history: List of all epoch metric dicts.
    """
    # Handle model wrapper vs plain nn.Module
    if hasattr(model, "backbone"):
        model_state = model.backbone.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "config": config,
        "metrics_history": metrics_history,
        "best_miou": best_miou,
        "best_bf1": best_bf1,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    strict: bool = True,
) -> dict:
    """Load a checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        model: If provided, load model weights. Handles wrapper vs plain module.
        optimizer: If provided, restore optimizer state.
        scheduler: If provided, restore scheduler state.
        strict: Whether to require exact key match in model state dict.

    Returns:
        The full checkpoint dict (always returned for access to epoch, metrics, etc.)

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"To start fresh, do not pass --resume. "
            f"To see available checkpoints, check experiments/runs/."
        )

    # Load to CPU first, then let model handle device placement
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if model is not None:
        target = model.backbone if hasattr(model, "backbone") else model
        target.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt
