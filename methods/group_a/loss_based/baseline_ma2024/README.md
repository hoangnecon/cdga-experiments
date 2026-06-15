# Boundary F1 Loss (Ma2024)

**Ma et al., TGRS 2024**: *"SAM-assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints"*

Official code: [`tmp/SSRS/SAM_RS/utils.py`](../../../../tmp/SSRS/SAM_RS/utils.py) (lines 394–449)

## Implementation Status

⚠️ **FAITHFUL TO OFFICIAL CODE** — This implementation reproduces the official code exactly, including implementation issues identified in the original. No bugs have been fixed beyond what is necessary to make the code runnable on GPU.

### Changes from official code (MINIMAL — only to make it run)

| Official (line) | Ours (line) | Rationale |
|-----------------|-------------|-----------|
| `class_map = pred.argmax(dim=1).cpu()` (L415) | `class_map = pred.argmax(dim=1)` (L30) | `.cpu()` causes device mismatch crash when data is on GPU. Removing it preserves gradient graph behavior (1). |
| `1 - gt` on LongTensor (L419) | `gt_float = gt.float()` then `1.0 - gt_float` (L35–37) | `F.max_pool2d` requires float input. Official code works on CPU with implicit casting; explicit on GPU. |

(1) `argmax` is still non-differentiable regardless of device — see Bug Analysis below.

---

## Identified Implementation Issues in Official Code

The following issues exist in the official implementation and are preserved in our faithful reproduction.

### Bug A: `argmax` is Non-Differentiable (CRITICAL for gradient flow)

**Official code** (lines 414–415):
```python
pred = torch.softmax(pred, dim=1)
class_map = pred.argmax(dim=1).cpu()
```

**Issue**: `torch.argmax` is a non-argmax differentiable operation. The gradient of `argmax` is zero everywhere. Therefore, the entire computation chain `logits → softmax → argmax → boundary_map → boundary_loss` has **zero gradient** with respect to logits.

**Scientific consequence**: The `bdy_loss` term contributes **zero gradient** to the model parameters. The model is effectively optimized by the `ce_loss` term alone. The boundary loss value in the total loss is purely a scalar monitor — it does not affect training.

**Why we don't fix this**: Replacing `argmax` with a differentiable alternative (e.g., soft predictions, Gumbel-softmax) would change the method. The result would no longer be comparable to the published Ma2024 results.

### Bug B: `1 − class_map` is Incorrect for Multiclass Segmentation

**Official code** (lines 422–424):
```python
pred_b = F.max_pool2d(1 - class_map, kernel_size=theta0, ...)
pred_b -= 1 - class_map
```

**Issue**: For binary segmentation (C=2), `class_map ∈ {0, 1}`, so `1 − class_map ∈ {0, 1}` — correct.
For multiclass segmentation (C > 2), `class_map ∈ {0, ..., C−1}`, so `1 − class_map` is negative for class ≥ 2, producing a meaningless boundary map.

**Scientific consequence**: On multiclass datasets (e.g., Vaihingen C=6, Potsdam C=6, LoveDA C=7), the predicted boundary map is mathematically invalid. The F1 score computed from it is unreliable.

**Why we don't fix this**: Fixing would require redesigning the boundary extraction for multiclass (e.g., per-class boundary maps, or semantic boundary detection). This would be a different method.

### Bug C: `.view(n, 2, -1)` Assumes Binary Spatial Dimensions

**Official code** (lines 434–437):
```python
gt_b = gt_b.view(n, 2, -1)
pred_b = pred_b.view(n, 2, -1)
```

**Issue**: The code reshapes a tensor of shape `(N, H, W)` into `(N, 2, H*W//2)`. This effectively **splits the image into two halves** (top half → channel 0, bottom half → channel 1) and computes Precision/Recall separately for each half.

This appears to be a bug: the code assumes `gt_b` has shape `(N, 2, H, W)` (binary one-hot boundary mask), but in practice `gt_b` is `(N, H, W)` (single-channel integer mask). The `.view(n, 2, -1)` operation does not raise an error when `H*W` is even; it silently produces a different computation than intended.

**Scientific consequence**: Precision, Recall, and BF1 are computed over two disjoint spatial halves of the image. The resulting F1 does not represent the boundary quality of the entire image.

**Why we don't fix this**: Fixing requires understanding what the authors intended (binary one-hot) vs what the code actually computes. We reproduce what was published and run.

---

## Net Effect on Training

Due to Bug A (non-differentiable `argmax`), the effective training loss for Ma2024 is:

$$L_{\text{effective}} = L_{\text{CE}} + \cancel{\lambda \cdot L_{\text{bdy}}} = L_{\text{CE}}$$

The boundary term adds a scalar value to the total loss but contributes zero gradient. Ma2024 results should therefore be **very close to baseline CE results** on all datasets. Any observed difference is due to:
1. Random seed variation
2. The scalar boundary term causing minor numerical differences in the loss value (which does not affect optimizer steps)

---

## Default Hyperparameters
| Parameter | Value | Source |
|-----------|-------|--------|
| `theta0` | 3 | Boundary extraction kernel size |
| `theta` | 5 | Extended boundary tolerance kernel size |
| `weight` | 0.1 | Coefficient in $L_{total} = L_{CE} + w \cdot L_{bdy}$ |

## Mathematical Formulation (as described in paper)
$$L_{bdy} = 1 - \text{BF1}, \quad \text{BF1} = \frac{2 \cdot P \cdot R}{P + R + \epsilon}$$

where $P$ and $R$ are precision and recall of predicted soft boundary against ground-truth boundary, with extended tolerance.

**Note**: The implementation of this formulation in the official code contains the issues documented above, so the actual computation differs from the mathematical description in the paper.