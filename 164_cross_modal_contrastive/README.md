# Cross-Modal Contrastive Learning for Stocks

This repository implements **Cross-Modal Contrastive Learning** for financial time series and associated textual data (e.g., news headlines, financial reports, or tweets). It adapts the core principles of OpenAI's CLIP (Contrastive Language-Image Pretraining) to the financial domain: **Contrastive Language-Timeseries Pretraining**.

## Core Concept

Price action does not happen in a vacuum. A sudden 5% drop on a chart ($x_{price}$) is often directly tied to a real-world event described in text ($x_{text}$), such as "Company misses earnings expectations."

Cross-Modal Contrastive Learning aligns these two different modes of information into a single, shared latent space.
- **Positive Pair**: A specific price chart window and the news headline that occurred *at the exact same time*.
- **Negative Pairs**: That same price chart window paired with random news headlines from other days or other assets.

By training the network to maximize the cosine similarity between the true (Price, Text) pairs while minimizing it for all other combinations in a batch, the model learns a phenomenally rich representation.

## Trading Advantages

- **Zero-Shot Event Search**: You can use a text query (e.g., "sudden flash crash") to search through millions of historical charts and find visually similar events without ever training a specific "flash crash" classifier.
- **Semantic Chart Understanding**: The time-series encoder learns what chart patterns actually *mean* in the real world, rather than just memorizing geometric shapes.
- **Contextual Signals**: By projecting current price action and current news into the same space, you can calculate their similarity. If the price is dropping but the news embedding says "bullish breakout", it might indicate anomalous market manipulation or an upcoming reversal.

## Project Structure

- `python/`: 
    - PyTorch implementation of the `TimeSeriesEncoder` (1D-CNN) and `TextEncoder` (LSTM/Transformer).
    - The symmetric `CLIPLoss` (InfoNCE across both modalities).
    - A synthetic dataset generator that pairs specific chart structures with specific tokenized text sequences.
- `rust/`: 
    - High-performance Rust library for real-time dual-encoder inference (processing both tick data and tokenized news feeds).
- `docs/`: Theoretical deep dive and implementation details.

## References

- Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, OpenAI, 2021).
