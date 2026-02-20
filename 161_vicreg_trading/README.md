# VICReg Trading: Variance-Invariance-Covariance Regularization

This repository implements **VICReg** (Variance-Invariance-Covariance Regularization) for self-supervised learning on financial time series data. VICReg is an SSL method that effectively prevents representation collapse without requiring negative samples, asymmetrical architectures (like BYOL), or clustering (like SwAV).

## Core Concept

VICReg operates on two augmented views of the same data, $x$ and $x'$, producing embeddings $z$ and $z'$. It uses three distinct loss terms to shape the embedding space:

1.  **Invariance (Sim)**: Minimizes the distance between $z$ and $z'$. The model learns that augmented versions of the same market window should represent the same underlying state.
2.  **Variance (Var)**: Forces the standard deviation of each dimension across the batch to be above a threshold $\gamma$ (usually 1.0). This prevents **Point Collapse** (all inputs mapping to the same vector).
3.  **Covariance (Cov)**: Minimizes the off-diagonal terms of the covariance matrix of $z$. This prevents **Dimension Collapse** (all dimensions becoming highly correlated), forcing the model to utilize its full embedding capacity.

## Trading Advantages

- **No Negatives Required**: Trading data is notoriously difficult to sample "negatives" for (is a 1% drop today "different" from a 1% drop a year ago?). VICReg avoids this ambiguity.
- **Robust Feature Diversity**: By explicitly regularizing the covariance matrix, VICReg ensures that the features learned (e.g., volatility, trend, momentum) are independent and non-redundant.
- **Stable Training**: Highly stable loss function compared to contrastive methods, making it suitable for volatile financial time series.

## Project Structure

- `python/`: PyTorch implementation of the VICReg model and training loop.
- `rust/`: High-performance Rust library for real-time feature extraction.
- `docs/`: Theoretical deep dive and implementation details.

## References

- Bardes, A., Ponce, J., & LeCun, Y. (2021). [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906).
