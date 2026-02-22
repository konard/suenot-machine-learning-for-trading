# Chapter 170: Prototypical Contrastive Learning (PCL)

## Overview

In the previous chapters, we focused on **Instance-level** contrastive learning (InfoNCE, SimCLR, CPC), where we compare one sample against others in a batch. While effective, this approach can be noisy and ignores the underlying global structure of the data.

**Prototypical Contrastive Learning (PCL)** bridges the gap between self-supervised contrastive learning and clustering. It encourages the model to learn features that not only distinguish individual instances but also align with **prototypes** (representative centroids) that capture the global semantic structure of the market.

## How it Works

1. **Feature Extraction**: Real-time market features are extracted using a CNN encoder.
2. **K-Means Clustering**: Periodically, the entire embedding space is clustered into $K$ prototypes using K-Means.
3. **ProtoNCE Loss**: For each sample, the model minimizes a loss that pulls it toward its assigned prototype while pushing it away from other prototypes.
4. **Hierarchical Learning**: Multiple levels of $K$ (e.g., $K=10, 50, 200$) can be used to capture structure at different scales (e.g., broad regimes vs. specific asset behaviors).

## Benefits for Trading

- **Noise Reduction**: Categorizing a price pattern into a "prototype" help filter out high-frequency noise that doesn't belong to any global regime.
- **Regime Identification**: Prototypes naturally correspond to market regimes (e.g., "High Volatility Bullish" or "Low Volatility Sideways").
- **Global Consistency**: Unlike standard contrastive learning which is limited by batch size, PCL enforces consistency across the entire dataset via prototypes.

## Project Structure

```
170_proto_contrastive/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py       # Base CNN Encoder
│   ├── pcl_loss.py    # ProtoNCE implementation
│   └── train.py       # Iterative clustering & contrast
└── rust/src/
    └── lib.rs         # Optimized centroid calculation
```
