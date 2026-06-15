"""
Component: Class-Directed Gradient Amplification (CDGA) Utilities
Location: shared/utils/cdga_utils.py

Handles:
- Boundary mask extraction (morphological)
- Distance map transform
- Spatial modulation mask construction
- CDGAHook implementation
"""
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_boundary(label: np.ndarray) -> np.ndarray:
    """Extract boundary pixels from label map (4-connectivity).

    Args:
        label: (H, W) integer label map.
    Returns:
        (H, W) boolean boundary mask.
    """
    boundary = np.zeros_like(label, dtype=bool)
    boundary[:-1, :] |= (label[:-1, :] != label[1:, :])
    boundary[1:, :]  |= (label[:-1, :] != label[1:, :])
    boundary[:, :-1] |= (label[:, :-1] != label[:, 1:])
    boundary[:, 1:]  |= (label[:, :-1] != label[:, 1:])
    return boundary


def compute_distance_map(label: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance transform from boundary.

    D_M(x,y) = min_{(u,v) in B} ||(x,y) - (u,v)||_2

    Args:
        label: (H, W) integer label map.
    Returns:
        (H, W) float32 distance map.
    """
    boundary = compute_boundary(label)
    # distance transform is 0 at boundary, positive inside/outside
    D_M = distance_transform_edt(~boundary).astype(np.float32)
    return D_M


def compute_modulation_mask(
    D_M: np.ndarray,
    sigma: float = 5.0,
    decay_fn: str = "gaussian"
) -> np.ndarray:
    """Compute spatial modulation mask S(x,y) in [0, 1].

    Args:
        D_M: (H, W) distance map.
        sigma: bandwidth parameter.
        decay_fn: 'gaussian' | 'linear' | 'step'.
    Returns:
        (H, W) float32 mask in [0, 1].
    """
    if decay_fn == "gaussian":
        S = np.exp(-D_M ** 2 / (2 * sigma ** 2))
    elif decay_fn == "linear":
        S = np.clip(1.0 - D_M / (3 * sigma), 0.0, 1.0)
    elif decay_fn == "step":
        S = (D_M <= sigma).astype(np.float32)
    else:
        raise ValueError(f"Unknown decay_fn: {decay_fn}")
    return S.astype(np.float32)


class CDGAHook:
    """Gradient hook for CDGA.

    Modulates gradient at a feature map during backprop.
    Supports two modes:
    1. 'class_directed' (CDGA): projects the gradient onto the correct class prototype in feature space.
    2. 'scalar': simple spatial amplification (mathematically equivalent to DW-CE).
    """

    def __init__(self, gamma: float = 10.0, mode: str = "class_directed"):
        self.gamma = gamma
        self.mode = mode
        self._mask = None
        self._labels = None
        self._head_weights = None
        self._handle = None
        self.grad_stats = {}

    def set_inputs(self, S_mask: torch.Tensor, labels: torch.Tensor, head_weights: torch.Tensor) -> None:
        """Set modulation inputs for current batch.

        Args:
            S_mask: (B, 1, H, W) spatial modulation mask.
            labels: (B, H, W) ground truth labels.
            head_weights: (C, C_f, 1, 1) or (C, C_f) classification head weights.
        """
        self._mask = S_mask
        self._labels = labels
        
        # Squeeze weights to shape (C, C_f) and detach
        if len(head_weights.shape) == 4:
            self._head_weights = head_weights.sum(dim=(2, 3)).detach()
        else:
            self._head_weights = head_weights.detach()

    def set_mask(self, S_mask: torch.Tensor) -> None:
        """Fallback for compatibility with scalar mode."""
        self._mask = S_mask

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """Modulate gradient. Called by autograd engine."""
        if self._mask is None:
            return grad

        B, C_f, H, W = grad.shape
        import torch.nn.functional as F
        
        # Resize mask to match feature map spatial dimensions
        mask_resized = F.interpolate(self._mask, size=(H, W), mode='bilinear', align_corners=False)

        if self.mode == "class_directed":
            if self._labels is None or self._head_weights is None:
                raise ValueError("labels and head_weights must be set for class_directed mode. Call set_inputs() first.")
            
            num_classes = self._head_weights.size(0)
            
            # Resize labels using nearest neighbor to preserve class indices
            labels_resized = F.interpolate(self._labels.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
            
            # Clamp labels to valid class range [0, C-1] for prototype lookup (handles ignore index)
            safe_labels = torch.clamp(labels_resized, 0, num_classes - 1)
            
            # Lookup correct class prototype per pixel: w_k shape (B, H, W, C_f)
            w_k = self._head_weights[safe_labels]
            w_k = w_k.permute(0, 3, 1, 2)  # shape (B, C_f, H, W)
            
            # Project grad onto w_k: proj = <grad, w_k> / ||w_k||^2 * w_k
            dot = (grad * w_k).sum(dim=1, keepdim=True)  # shape (B, 1, H, W)
            norm_sq = (w_k * w_k).sum(dim=1, keepdim=True) + 1e-8  # shape (B, 1, H, W)
            proj = (dot / norm_sq) * w_k  # shape (B, C_f, H, W)
            
            # Only modulate at non-ignored pixels
            valid_mask = (labels_resized != 255).unsqueeze(1).float()
            
            # Modulate gradient: add gamma * S * proj to original gradient
            modulated = grad + self.gamma * mask_resized * proj * valid_mask
        else:
            # Simple scalar modulation
            modulated = grad * (1.0 + self.gamma * mask_resized)

        # Log stats
        with torch.no_grad():
            boundary_region = (mask_resized > 0.5)
            interior_region = ~boundary_region
            gnorm = grad.norm(dim=1, keepdim=True)
            mnorm = modulated.norm(dim=1, keepdim=True)

            if boundary_region.any():
                self.grad_stats["orig_boundary"] = gnorm[boundary_region.expand_as(gnorm)].mean().item()
                self.grad_stats["mod_boundary"] = mnorm[boundary_region.expand_as(mnorm)].mean().item()
            if interior_region.any():
                self.grad_stats["orig_interior"] = gnorm[interior_region.expand_as(gnorm)].mean().item()
                self.grad_stats["mod_interior"] = mnorm[interior_region.expand_as(mnorm)].mean().item()

        return modulated

    def attach(self, feature_map: torch.Tensor) -> object:
        """Attach hook to feature map."""
        self._handle = feature_map.register_hook(self)
        return self._handle

    def detach(self) -> None:
        """Remove hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
