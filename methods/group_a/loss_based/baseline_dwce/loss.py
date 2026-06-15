"""
Method: Distance-Weighted CE (DW-CE)
Component: Loss Definition
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn

class DWCELoss(nn.Module):
    """Distance-Weighted Cross Entropy Loss."""
    def __init__(self, gamma: float = 10.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        loss_pixel = self.ce_fn(logits, targets)  # (B, H, W)
        
        # boundary_mask is (B, 1, H, W), we squeeze the channel dimension
        weight = 1.0 + self.gamma * boundary_mask.squeeze(1)  # (B, H, W)
        weighted_loss = loss_pixel * weight
        
        valid_mask = (targets != self.ignore_index).float()
        total_valid = valid_mask.sum()
        
        if total_valid == 0:
            return weighted_loss.sum() * 0.0
            
        return weighted_loss.sum() / total_valid
