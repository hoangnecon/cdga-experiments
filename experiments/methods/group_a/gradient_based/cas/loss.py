"""CAS: Conflict-Aware Scaling. Standard CE loss."""
import torch.nn as nn

def get_loss_fn(ignore_index: int = 255) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
