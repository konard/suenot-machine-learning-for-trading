# Cross-Modal Contrastive Learning for Trading

This chapter implements **Cross-Modal Contrastive Learning** for financial time series and associated textual data (news headlines, financial reports, tweets). It adapts the core principles of OpenAI's CLIP (Contrastive Language-Image Pretraining) to the financial domain: **Contrastive Language-Timeseries Pretraining**.

Supports both **stock market** (AAPL, S&P 500) and **cryptocurrency** (Bybit: BTCUSDT, ETHUSDT) data.

## Core Concept

Price action does not happen in a vacuum. A sudden 5% drop on a chart ($x_{price}$) is often directly tied to a real-world event described in text ($x_{text}$), such as "Company misses earnings expectations" or "Bybit BTCUSDT flash crash triggered by whale liquidation."

Cross-Modal Contrastive Learning aligns these two different modes of information into a single, **shared latent space**.

- **Positive Pair**: A specific price chart window and the news headline that occurred *at the exact same time*.
- **Negative Pairs**: That same price chart window paired with random news headlines from other days or other assets.

By training the network to maximize the cosine similarity between the true (Price, Text) pairs while minimizing it for all other combinations in a batch, the model learns a rich semantic representation of the market.

### Mathematical Formulation

Given a batch of $N$ pairs, the similarity matrix $S$ is:

$$S_{i,j} = \frac{v_{price}^{(i)} \cdot v_{text}^{(j)}}{||v_{price}^{(i)}|| \cdot ||v_{text}^{(j)}||}$$

The symmetric CLIP loss is:

$$\mathcal{L}_{CLIP} = \frac{1}{2} \left[ \frac{1}{N}\sum_{i} \mathcal{L}_{P \rightarrow T}^{(i)} + \frac{1}{N}\sum_{j} \mathcal{L}_{T \rightarrow P}^{(j)} \right]$$

where each term is a temperature-scaled cross-entropy over the similarity matrix rows/columns.

## Trading Advantages

- **Zero-Shot Event Search**: Use a text query (e.g., "sudden flash crash on Bybit") to search through millions of historical charts and find visually similar events without training a specific classifier.
- **Semantic Chart Understanding**: The time-series encoder learns what chart patterns actually *mean* in the real world, rather than just memorizing geometric shapes.
- **Divergence Trading Signal**: By projecting current price action and current news into the same space, you can calculate their similarity divergence. If the price is dropping but the news embedding says "bullish breakout", it signals potential manipulation or an upcoming reversal.
- **Cross-Market Transfer**: The shared latent space allows comparing patterns across stocks (AAPL) and crypto (BTCUSDT), finding structural similarities in different markets.

## Project Structure

```
164_cross_modal_contrastive/
├── README.md                  # This file (English)
├── README.ru.md               # Russian translation
├── readme.simple.md           # Simplified explanation (English)
├── readme.simple.ru.md        # Simplified explanation (Russian)
├── python/
│   ├── model.py               # TimeSeriesEncoder, TextEncoder, CLIPLoss, SymmetricCLIPLoss
│   ├── data.py                # Data generation: synthetic, Bybit crypto, stock market
│   ├── train.py               # Training script (overfit test + market data mode)
│   ├── evaluate.py            # Zero-shot retrieval evaluation
│   └── requirements.txt       # Python dependencies
├── rust/
│   ├── Cargo.toml             # Rust dependencies
│   └── src/
│       └── lib.rs             # High-performance embedding search engine
└── docs/
    └── ru/
        └── theory.md          # Theoretical deep dive (Russian)
```

## Quick Start

### Python

```bash
cd python
pip install -r requirements.txt

# Mode 1: Static overfit test (verifies architecture)
python train.py --mode overfit
python evaluate.py --mode overfit

# Mode 2: Train on Bybit crypto + stock market data
python train.py --mode market
python evaluate.py --mode market
```

### Rust

```bash
cd rust
cargo test
```

## Data Sources

### Bybit Cryptocurrency (Simulated)
- **BTCUSDT**: Bitcoin perpetual futures with events: pump, crash, short squeeze, rally, dump
- **ETHUSDT**: Ethereum perpetual futures with similar event taxonomy

### Stock Market (Simulated)
- **AAPL**: Apple Inc. with events: earnings beat, Fed rate hike, sector rally, CEO scandal

All data is generated synthetically within this chapter to ensure reproducibility and self-containment.

## Architecture

### TimeSeriesEncoder (1D-CNN)
- 3 convolutional blocks with BatchNorm + ReLU
- Adaptive average pooling → projection head
- Input: `(B, 1, 128)` → Output: `(B, 32)`

### TextEncoder (Embedding + Mean Pooling)
- Learnable token embeddings with padding mask
- LayerNorm → projection head
- Input: `(B, 8)` → Output: `(B, 32)`

### Loss Functions
1. **CLIPLoss**: CosineEmbeddingLoss with explicit positive/negative pairs
2. **SymmetricCLIPLoss**: Full NxN InfoNCE with learnable temperature (recommended)

## References

- Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, OpenAI, 2021). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Oord, A. van den, et al. "Representation Learning with Contrastive Predictive Coding" (CPC, 2018). [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
