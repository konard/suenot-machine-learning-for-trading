# Chapter 177: Hierarchical Federated Learning for Trading

## Overview

As federated trading networks grow to global scales, the "Star" topology (where every client connects to one cloud server) becomes a bottleneck. High latency between New York, London, and Tokyo can stall model convergence.

**Hierarchical Federated Learning** (HFL) solves this by mirroring the structure of global finance:
1. **Edge level**: Individual trading bots or institutional desks perform local training.
2. **Regional level**: Regional hubs (e.g., a London office) aggregate updates from nearby clients.
3. **Global level**: The global cloud server aggregates regional summaries to form the master model.

## Key Advantages
- **Communication Efficiency**: Significantly reduces the number of long-distance sync rounds.
- **Privacy at Scales**: Data and semi-aggregated models stay within geographical boundaries longer.
- **Improved Convergence**: Regional nodes can capture regional alpha (e.g., local exchange idiosyncrasies) before merging them into the global knowledge pool.

## Project Structure

```
177_hierarchical_fl_trading/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py            # Base Neural Network
│   ├── hfl_core.py         # Multi-tier aggregation logic
│   └── train.py            # HFL vs. Flat FL simulation
└── rust/src/
    └── lib.rs              # Optimized multi-tensor aggregator
```
