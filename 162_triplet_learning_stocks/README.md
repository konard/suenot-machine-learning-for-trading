# Triplet Learning for Stocks

This repository implements **Triplet Learning** for self-supervised feature extraction on financial time series data. Instead of grouping all data (like clustering) or relying solely on positive pairs (like BYOL), Triplet Learning explicitly defines relative relationships between market regimes using Anchor, Positive, and Negative sequences.

## Core Concept

Triplet Learning operates on a tuple of three inputs:
1.  **Anchor ($x_a$)**: A reference market window.
2.  **Positive ($x_p$)**: An augmented version of the Anchor, OR a market window known to represent the exact same underlying market regime (e.g., another segment of the same bull run).
3.  **Negative ($x_n$)**: A market window known to represent a definitively different regime (e.g., a flash crash).

The **Triplet Margin Loss** forces the neural network to learn an embedding space where the distance between the Anchor and the Positive is strictly smaller than the distance between the Anchor and the Negative, plus a predefined margin $m$.

$$ \mathcal{L} = \max(0, d(z_a, z_p) - d(z_a, z_n) + m) $$

## Trading Advantages

- **Explicit Separation**: By explicitly defining Negatives, the model learns fine-grained boundaries between what constitutes "similar" vs "different" market conditions.
- **Relative Distances**: It doesn't force embeddings to absolute coordinate spaces; it only cares about relative distances. This makes distance calculations (like K-Nearest Neighbors) extremely reliable in production.
- **No Global Contrastive Overhead**: Unlike SimCLR, which requires massive batch sizes to act as negatives, Triplet Loss only requires carefully mined triplets.

## Project Structure

- `python/`: PyTorch implementation of the `TripletNet`, `CNN1DEncoder`, and the training loop featuring a custom triplet data generator.
- `rust/`: High-performance Rust library for real-time feature extraction and Euclidean distance calculation for nearest neighbor matching.
- `docs/`: Theoretical deep dive and implementation details.

## References

- Concept derived from FaceNet: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).
- Applied to financial data through Contrastive learning methodologies.
