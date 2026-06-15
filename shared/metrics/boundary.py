"""
Component: Metrics — Boundary (BF1@k, Hausdorff)
Location: shared/metrics/boundary.py

Ref:
    - rules/STRUCTURE.md
    - rules/CONVENTIONS.md Section 5
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def _get_boundary_mask(label: np.ndarray) -> np.ndarray:
    """Extract boundary pixels via 4-connectivity neighbor check.

    A pixel is on the boundary if any 4-neighbor has a different class label.

    Args:
        label: (H, W) integer label map.

    Returns:
        (H, W) boolean mask, True at boundary pixels.
    """
    boundary = np.zeros_like(label, dtype=bool)
    boundary[:-1, :] |= (label[:-1, :] != label[1:, :])   # down
    boundary[1:, :]  |= (label[:-1, :] != label[1:, :])   # up
    boundary[:, :-1] |= (label[:, :-1] != label[:, 1:])   # right
    boundary[:, 1:]  |= (label[:, :-1] != label[:, 1:])   # left
    return boundary


def compute_boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    dilation_widths: list[int] = [3, 5],
    ignore_index: int = 255,
) -> dict[str, float]:
    """Compute Boundary F1 score at multiple dilation widths.

    BF1@k: Boundary F1 with k-pixel symmetric dilation tolerance.
    Standard for ISPRS boundary quality evaluation.

    Formula:
        Precision = |pred_boundary ∩ dilated(gt_boundary)| / |pred_boundary|
        Recall    = |gt_boundary ∩ dilated(pred_boundary)| / |gt_boundary|
        BF1@k     = 2 * P * R / (P + R)

    Args:
        pred: (H, W) or (N, H, W) predicted class map.
        target: Same shape, ground-truth.
        dilation_widths: List of k values to compute BF1@k.
        ignore_index: Pixels to exclude from evaluation.

    Returns:
        dict with keys 'bf1_{k}' for each k in dilation_widths.
        Example: {'bf1_3': 0.6321, 'bf1_5': 0.7012}
    """
    # Handle batch dimension
    if pred.ndim == 3:
        results_per_image = [
            compute_boundary_f1(pred[i], target[i], dilation_widths, ignore_index)
            for i in range(pred.shape[0])
        ]
        # Average across images
        out = {}
        for k in dilation_widths:
            key = f"bf1_{k}"
            out[key] = float(np.mean([r[key] for r in results_per_image]))
        return out

    # Single image
    ignore_mask = target == ignore_index

    pred_b = _get_boundary_mask(pred)
    gt_b = _get_boundary_mask(target)

    # Exclude ignore regions from boundary
    pred_b[ignore_mask] = False
    gt_b[ignore_mask] = False

    results = {}
    for k in dilation_widths:
        struct = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)

        gt_dilated = binary_dilation(gt_b, structure=struct)
        pred_dilated = binary_dilation(pred_b, structure=struct)

        # Precision: fraction of predicted boundary pixels that hit GT boundary
        tp_prec = (pred_b & gt_dilated).sum()
        precision = float(tp_prec) / (pred_b.sum() + 1e-10)

        # Recall: fraction of GT boundary pixels that are covered by prediction
        tp_rec = (gt_b & pred_dilated).sum()
        recall = float(tp_rec) / (gt_b.sum() + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        results[f"bf1_{k}"] = float(f1)

    return results


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Compute (percentile) Hausdorff Distance between boundary sets.

    Args:
        pred: (H, W) predicted class map.
        target: (H, W) ground-truth class map.
        percentile: Use 95th percentile instead of max for robustness.

    Returns:
        HD in pixels. Lower is better. Returns inf if one map has no boundary.
    """
    pred_b = _get_boundary_mask(pred)
    gt_b = _get_boundary_mask(target)

    if not pred_b.any() or not gt_b.any():
        return float("inf")

    # Distance from each GT boundary pixel to nearest pred boundary pixel
    dist_pred_to_gt = distance_transform_edt(~gt_b)
    dist_gt_to_pred = distance_transform_edt(~pred_b)

    d1 = dist_pred_to_gt[pred_b]  # Distances from pred boundary to GT boundary
    d2 = dist_gt_to_pred[gt_b]   # Distances from GT boundary to pred boundary

    hd = max(
        np.percentile(d1, percentile),
        np.percentile(d2, percentile),
    )
    return float(hd)
