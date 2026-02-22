# Chapter 171: Federated Averaging (FedAvg)

## Overview

**Federated Averaging (FedAvg)** is the cornerstone algorithm of Federated Learning. In the context of trading, it allows multiple quantitative desks or institutions to train a powerful "global" model without ever sharing their proprietary trading data or client sensitive information.

Instead of moving data to the model (Centralized Training), we move the model to the data (Decentralized Training).

## The FedAvg Process

1. **Initialization**: A central server initializes the global model weights $w_0$.
2. **Communication Round** $t$:
   - **Selection**: The server selects a subset of clients.
   - **Broadcast**: The server sends the current global weights $w_t$ to these clients.
   - **Local Update**: Each client $k$ performs several epochs of SGD on their *local* data to compute new weights $w_{t+1}^k$.
   - **Aggregation**: Clients send their updated weights back. The server computes the new global weights as a weighted average:
     $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k$$
     where $n_k$ is the number of samples on client $k$.

## Why it Matters for Finance

- **Data Privacy**: Alpha signals are the most valuable secrets in trading. FedAvg ensures that raw data stays behind the firewall.
- **Regulatory Compliance**: Helps comply with GDPR and other data localization laws while still benefiting from "global" intelligence.
- **Collaborative Research**: Multiple entities can improve their local performance by learning from the shared trends discovered on other datasets.

## Project Structure

```
171_fedavg_trading/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py           # Shared Neural Network
│   ├── federated_core.py  # Aggregation logic
│   └── train.py           # Federated simulation
└── rust/src/
    └── lib.rs             # High-speed weight aggregation
```
