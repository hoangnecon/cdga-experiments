# Class-Directed Gradient Amplification (CDGA)

CDGA is our proposed gradient-engineering method for boundary-aware remote sensing semantic segmentation. It intercepts intermediate feature representations before the classification head and modulates their backward gradients spatially to amplify boundary signals without changing inference architecture or speed.

## Mathematical Formulation
The gradient modulation at pixel $(x, y)$ of the feature map $F$ is formulated as:
$$\hat{G}(x, y) = G(x, y) \cdot (1 + \gamma S(x, y))$$

where:
1. $G(x, y) = \frac{\partial \mathcal{L}_{CE}}{\partial F(x, y)}$ is the original backpropagated feature gradient.
2. $S(x, y) = \exp\left(-\frac{D_M(x,y)^2}{2\sigma^2}\right)$ is the spatial boundary modulation mask.
3. $\gamma$ is the amplification factor (default 10).

## Default Hyperparameters
- `gamma`: 10
- `sigma`: 5.0
- `decay_fn`: "gaussian"
- `hook_layer`: "last" (attaches to the feature representation immediately preceding the classification head)
- `mask_source`: "gt_dilation"
