# Baseline Cross-Entropy (CE)

This is the standard baseline using vanilla Cross Entropy loss on dense semantic predictions.

## Mathematical Formulation
The loss is computed as:
$$L_{CE} = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid x_i)$$

where $P(y_i \mid x_i)$ is the predicted probability for the ground-truth class $y_i$ at pixel $i$.

## Default Hyperparameters
No method-specific hyperparameters are used.
