"""
Method: Boundary F1 Loss (Ma2024)
Component: Loss Definition (Official Paper Implementation)
Ref: 
    - tmp/SSRS/SAM_RS/utils.py (dòng 394)
    - rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    """Official Boundary F1 Loss from Ma et al. (TGRS 2024)."""
    def __init__(self, theta0: int = 3, theta: int = 5) -> None:
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - pred: output from model (before softmax), shape (N, C, H, W)
            - gt: ground truth map, shape (N, H, W)
        Return:
            - boundary loss, averaged over mini-batch
        """
        n, _, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1)  # Keep on the same device as pred

        # boundary map
        # Convert to float to satisfy F.max_pool2d type constraints
        gt_float = gt.float()
        gt_b = F.max_pool2d(
            1.0 - gt_float, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1.0 - gt_float

        class_map_float = class_map.float()
        pred_b = F.max_pool2d(
            1.0 - class_map_float, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1.0 - class_map_float

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
