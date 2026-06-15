"""
Method: CDGA (Class-Directed Gradient Amplification)
Component: Loss Definition
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn

class CDGALoss(nn.Module):
    """CDGA loss wrapper (Cross Entropy)."""
    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
