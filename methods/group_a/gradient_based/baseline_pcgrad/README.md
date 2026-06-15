# PCGrad Baseline (Projecting Conflicting Gradients)

PCGrad resolves gradient conflicts between multiple objectives (in our case, region learning and boundary learning) by projecting conflicting gradients onto each other's normal planes.

## Mathematical Formulation
Given the region gradient $g_{region}$ and the boundary gradient $g_{boundary}$:
1. Compute their cosine similarity: $d = g_{region} \cdot g_{boundary}$.
2. If $d < 0$ (conflict), project each gradient onto the other's normal plane:
   $$g_{region} = g_{region} - \frac{g_{region} \cdot g_{boundary}}{\|g_{boundary}\|^2} g_{boundary}$$
   $$g_{boundary} = g_{boundary} - \frac{g_{boundary} \cdot g_{region}}{\|g_{region}\|^2} g_{region}$$
3. The final gradient is the sum: $g_{final} = g_{region} + g_{boundary}$.
4. If $d \geq 0$ (no conflict), the final gradient is $g_{final} = g_{region} + g_{boundary}$.

The region objective is standard Cross Entropy, and the boundary objective is Boundary Cross Entropy (CE computed only on boundary pixels).

## Default Hyperparameters
No method-specific hyperparameters are used.
