# Chapter 178: Asynchronous Federated Learning for Trading

## Overview

In the previous chapters, we used synchronous aggregation (FedAvg). While robust, synchronous FL has a fatal flaw in low-latency trading: it waits for the slowest participant (the "straggler").

**Asynchronous Federated Learning** (AFL) allows the central server to update the global model as soon as any single client's update arrives.

## The Challenge: Model Staleness
In AFL, a client might start training on version $N$ of the global model, but by the time their update arrives, the server has already moved to version $N+10$. This is called **Staleness**. If we treat outdated updates the same as fresh ones, the model will diverge.

## Our Solution: Staleness-Aware Aggregation
We will implement an aggregation rule that automatically down-weights "stale" updates using a decay function:
- **Fresh updates**: High influence on the global model.
- **Outdated updates**: Low influence, used only to preserve long-term patterns.

## Project Structure

```
178_asynchronous_fl_trading/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py            # Base Neural Network
│   ├── afl_core.py         # Asynchronous server logic
│   └── train.py            # AFL vs. Sync FL simulation
└── rust/src/
    └── lib.rs              # Optimized decay coefficient engine
```
