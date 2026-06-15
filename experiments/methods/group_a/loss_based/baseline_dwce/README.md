# Distance-Weighted Cross-Entropy (DW-CE)

Distance-Weighted Cross-Entropy modulates the cross entropy loss at each pixel by a factor proportional to its spatial distance to the nearest boundary.

## Mathematical Formulation
The loss is formulated as:
$$L_{DW-CE} = \frac{1}{\sum_{i} \mathbb{1}(y_i \neq \text{ignore})} \sum_{i: y_i \neq \text{ignore}} \left(1 + \gamma S(x_i, y_i)\right) L_{CE}(i)$$

where $S(x_i, y_i)$ is the boundary modulation mask computed as:
$$S(x, y) = \exp\left(-\frac{D_M(x,y)^2}{2\sigma^2}\right)$$

with $D_M(x,y)$ representing the Euclidean distance to the nearest boundary.

## Default Hyperparameters
- `gamma`: 10
- `sigma`: 5.0
- `decay_fn`: "gaussian"
