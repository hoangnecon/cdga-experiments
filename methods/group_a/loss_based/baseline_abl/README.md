# Active Boundary Loss (ABL)

**Wang et al., AAAI 2022**: *"Active Boundary Loss for Semantic Segmentation"*

## Implementation Status

✅ **FAITHFUL** — This implementation follows the official code ([`tmp/active-boundary-loss/abl.py`](../../../../tmp/active-boundary-loss/abl.py)) line-by-line. No deviations.

### Key components verified against official code:

1. **Adaptive epsilon PDB detection** (official lines 65–87): ϵ starts at 1e-5, increases via `eps *= 1.2` in a while-loop until the number of KL-divergent pixels ≤ 1% of image pixels. 3×3 dilation applied afterward.

2. **9-position direction prediction** (official lines 103–163): 8 neighbor directions + center (index 8). Direction GT = `argmin(distance_map)` over all 9 positions.

3. **Center-exclusion filtering** (official lines 155–156): Pixels where the center itself is the argmin (i.e., those already on the GT boundary) are excluded from the direction loss.

4. **KL divergence**: `KL(neighbor || center) = softmax(neighbor) * (log_softmax(neighbor) − log_softmax(center))`, with neighbor logits detached (official line 141).

5. **Distance weighting**: `Λ(M) = clamp(M, max=20) / 20` (official line 40).

## Mathematical Formulation

$$L_{ABL} = \frac{1}{|K'|} \sum_{k \in K'} \Lambda(M_k) \cdot \text{CE}(D_k^p, D_k^g)$$

where:
- $K'$ = PDB pixels after center-exclusion filtering
- $D_k^p$ = KL-based predicted direction distribution (8-way)
- $D_k^g$ = ground-truth direction = $\arg\min_{j \in \{0,...,8\}} M_{k+\Delta_j}$
- $\Lambda(M_k) = \min(M_k, 20) / 20$

## Default Hyperparameters
| Parameter | Value | Official source |
|-----------|-------|----------------|
| `max_N_ratio` | 0.01 (1%) | `max_N_ratio = 1/100` (line 33) |
| `max_clip_dist` | 20.0 | `max_clip_dist = 20.` (line 33) |
| `is_detach` | True | `isdetach=True` (line 33) |
| `weight` | 1.0 | Coefficient in $L_{total} = L_{CE} + w \cdot L_{ABL}$ |