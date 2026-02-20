# Temporal Contrastive Learning for Stocks

This repository implements **Temporal Contrastive Learning (TCL)** for self-supervised feature extraction on financial time series data. Instead of generating artificial permutations or noise to create "positive" pairs (like SimCLR or BYOL), TCL leverages the natural arrow of time.

## Core Concept

In financial markets, the underlying "regime" or state changes slowly relative to minor tick-by-tick noise. Therefore, a time window $x_t$ and its immediate temporal neighbor $x_{t+k}$ (where $k$ is small) share the same underlying latent market state. 

TCL uses this structural property:
- **Positive Pair**: $(x_t, x_{t+k})$ — Architecturally treated as different views of the *same* underlying regime.
- **Negative Pairs**: $(x_t, x_{rand})$ — Time windows randomly sampled from distant points in the dataset, representing *different* regimes.

The network is trained using an **InfoNCE** (Normalized Temperature-scaled Cross Entropy) loss, but instead of the positives being artificially augmented images, they are naturally occurring adjacent time slices.

## Trading Advantages

- **No Artificial Distortion**: Financial data is continuous. Artificially injecting random noise or reversing signals to create augmentations can destroy crucial financial meaning. TCL avoids this by using the raw, unaltered data sequence.
- **Temporal Smoothness**: Embeddings learned via TCL evolve smoothly in the latent space over time. This makes them exceptionally useful for Hidden Markov Models (HMMs) or regime-switching classifiers, as the model naturally tracks macro-state evolution.
- **Trend Invariant**: By contrasting local adjacency with global distances, the model intrinsically learns what makes a specific local trend unique compared to the rest of the historical dataset.

## Project Structure

- `python/`: PyTorch implementation of the `CNN1DEncoder`, the target projection head, the InfoNCE-based `TemporalContrastiveLoss`, and a custom training loop that samples adjacent temporal windows.
- `rust/`: High-performance Rust library for real-time temporal feature extraction in production trading systems.
- `docs/`: Theoretical deep dive and implementation details.

## References

- Concept inspired by Contrastive Predictive Coding (CPC) and modern Temporal Contrastive Learning frameworks for video and continuous streams.
