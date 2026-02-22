# Chapter 172: FedProx for Finance

## Overview

In the previous chapter, we explored **FedAvg**, the simplest form of Federated Learning. While FedAvg works well with uniform data, it can struggle or even diverge in heterogeneous environments.

**FedProx** (Li et al., 2018) is designed for decentralized networks where:
1. **Data is Non-IID**: Different trading desks have radically different portfolios and risk profiles.
2. **Compute is Heterogeneous**: Some clients are high-performance servers, while others are low-power edge devices.

## The Proximal Advantage

FedProx introduces a **Proximal Term** to the local objective function on each client:
$$min_w h_k(w; w_t) = F_k(w) + \frac{\mu}{2} \|w - w_t\|^2$$

Where:
- $F_k(w)$ is the local loss (e.g., MSE).
- $w_t$ is the global model from the server.
- $\mu$ is a hyperparameter that controls the "stickiness" to the global model.

### Why this helps Trading
Standard FedAvg lets "noisy" or "outlier" clients pull the global model too far in their direction during local epochs. FedProx forces clients to stay reasonably close to the global consensus, preventing local over-optimization on idiosyncratic market noise.

## Project Structure

```
172_fedprox_finance/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py           # Shared Neural Network
│   ├── fedprox_core.py    # Proximal loss implementation
│   └── train.py           # Heterogeneous simulation
└── rust/src/
    └── lib.rs             # Optimized L2 distance calculator
```
