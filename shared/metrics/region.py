"""
Component: Metrics — Region (mIoU, mF1, OA)
Location: shared/metrics/region.py

Ref:
    - rules/STRUCTURE.md
    - rules/CONVENTIONS.md Section 5
"""
import numpy as np


def compute_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float | dict]:
    """Compute mean Intersection over Union.

    Args:
        pred: (N, H, W) or (H, W) predicted class map, integer.
        target: Same shape as pred, ground-truth class map.
        num_classes: Number of valid classes.
        ignore_index: Pixel value to ignore (e.g. void/boundary in ISPRS).

    Returns:
        {
            'miou': float in [0, 1],
            'per_class_iou': {class_id: float},
        }
    """
    pred = pred.flatten()
    target = target.flatten()

    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    per_class_iou: dict[int, float] = {}
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        union = tp + fp + fn
        per_class_iou[c] = float(tp) / float(union) if union > 0 else float("nan")

    valid_ious = [v for v in per_class_iou.values() if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {"miou": miou, "per_class_iou": per_class_iou}


def compute_mf1(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mean F1 score (macro-averaged over classes).

    Returns:
        float in [0, 1].
    """
    pred = pred.flatten()
    target = target.flatten()

    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    f1_scores = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if (target == c).sum() > 0:  # Only include classes that appear in GT
            f1_scores.append(float(f1))

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_oa(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255,
) -> float:
    """Compute Overall Accuracy (pixel-level).

    Returns:
        float in [0, 1].
    """
    pred = pred.flatten()
    target = target.flatten()
    valid_mask = target != ignore_index
    correct = (pred[valid_mask] == target[valid_mask]).sum()
    total = valid_mask.sum()
    return float(correct) / float(total) if total > 0 else 0.0
