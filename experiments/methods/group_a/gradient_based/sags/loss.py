"""
Method: SAGS (Spatially-Aware Gradient Surgery)
Component: Loss Definition — standard CE

SAGS modifies gradients in the backward pass, not the loss function.
We use plain CE loss here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(ignore_index: int = 255) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
