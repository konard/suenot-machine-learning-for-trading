# Chapter 350: Subgraph Mining for Trading — Discovering Hidden Patterns in Financial Networks

## Overview

Financial markets form complex networks: assets are connected through correlations, companies through supply chains, traders through order flow. **Subgraph mining** discovers recurring structural patterns (motifs) in these networks that can predict price movements, regime changes, and trading opportunities.

Traditional graph analysis looks at node-level or edge-level features. Subgraph mining goes deeper — it finds **frequent substructures** that appear across the graph or across time, revealing hidden market dynamics invisible to standard methods.

## Trading Strategy

**Core Idea:** Build dynamic financial graphs from market data and mine frequent subgraph patterns. When specific patterns appear, they signal trading opportunities:

1. **Correlation Networks:** Assets as nodes, correlations as edges. Mining subgraphs reveals sector rotations and contagion patterns
2. **Order Flow Graphs:** Traders/addresses as nodes, transactions as edges. Patterns reveal accumulation, distribution, whale movements
3. **Temporal Graphs:** Same network across time. Pattern changes signal regime shifts

**Edge:** Subgraph patterns capture higher-order dependencies that correlation matrices and pairwise analysis miss. A triangle of strong correlations has different meaning than a chain.

## Key Concepts

### What is a Subgraph?

A subgraph is a smaller graph contained within a larger graph. For example, in a network of 100 cryptocurrencies connected by correlations:
- A **triangle** (3 nodes, 3 edges) might represent a sector (BTC-ETH-BNB)
- A **star** (1 hub connected to many) might represent a market leader
- A **clique** (fully connected subgroup) represents highly correlated assets

### Subgraph Mining Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| gSpan | Frequent subgraph | Finding common patterns |
| FSG | Frequent subgraph | Sparse graphs |
| SUBDUE | Compression-based | Finding informative patterns |
| GraMi | Approximate | Large graphs |
| Neural (GNN) | Learning-based | Prediction tasks |

### Financial Graph Construction

```
Market Data → Graph Construction → Subgraph Mining → Trading Signals
     ↓              ↓                    ↓               ↓
  OHLCV       Correlation/Flow       Patterns      Entry/Exit
  Order Book  Threshold edges      Frequency       Position sizing
  Transactions  Dynamic updates    Support         Risk management
```

## Technical Specification

### Graph Types for Trading

#### 1. Correlation Network
```
Nodes: Assets (cryptocurrencies)
Edges: Correlation > threshold (e.g., 0.7)
Edge weight: Correlation strength
Temporal: Rolling window (e.g., 24h, 7d)
```

#### 2. Order Flow Graph
```
Nodes: Wallet addresses / Trader IDs
Edges: Transactions between nodes
Edge weight: Volume / Frequency
Temporal: Transaction timestamps
```

#### 3. Lead-Lag Network
```
Nodes: Assets
Edges: Granger causality / Transfer entropy
Edge weight: Predictive strength
Direction: Leader → Follower
```

### Subgraph Pattern Types

| Pattern | Structure | Trading Signal |
|---------|-----------|----------------|
| Triangle | 3 connected assets | Sector strength |
| Star | Hub + spokes | Market leader influence |
| Chain | Linear path | Information cascade |
| Clique-4+ | Fully connected 4+ | Strong regime |
| Bipartite | Two groups | Sector rotation |

### Mining Algorithm: gSpan for Financial Networks

```
1. Build graph G from market data
2. Find frequent edges (support > min_sup)
3. Extend patterns using DFS code
4. Prune non-frequent patterns
5. Output frequent subgraph patterns
```

### Signal Generation

```
For each trading period:
    1. Build current graph G_t from latest data
    2. Match known patterns against G_t
    3. Calculate pattern scores:
       - Frequency change (new pattern appearance)
       - Completeness (partial vs full match)
       - Centrality (pattern position in network)
    4. Generate signals based on pattern semantics
```

## Implementation Architecture (Rust)

```
src/
├── lib.rs              # Library entry point
├── graph/
│   ├── mod.rs          # Graph module
│   ├── types.rs        # Node, Edge, Graph types
│   ├── builder.rs      # Graph construction from data
│   └── algorithms.rs   # Graph algorithms
├── mining/
│   ├── mod.rs          # Mining module
│   ├── gspan.rs        # gSpan algorithm
│   ├── pattern.rs      # Pattern representation
│   └── matcher.rs      # Pattern matching
├── trading/
│   ├── mod.rs          # Trading module
│   ├── signals.rs      # Signal generation
│   ├── strategy.rs     # Trading strategy
│   └── backtest.rs     # Backtesting engine
├── data/
│   ├── mod.rs          # Data module
│   ├── bybit.rs        # Bybit API client
│   ├── types.rs        # Market data types
│   └── cache.rs        # Data caching
└── utils/
    ├── mod.rs          # Utilities
    ├── stats.rs        # Statistical functions
    └── config.rs       # Configuration
```

## Key Metrics

### Pattern Quality
- **Support:** Frequency of pattern occurrence
- **Significance:** Statistical significance vs random
- **Stability:** Persistence across time windows
- **Predictive Power:** Correlation with future returns

### Trading Performance
- **Sharpe Ratio:** Risk-adjusted returns
- **Sortino Ratio:** Downside-adjusted returns
- **Maximum Drawdown:** Largest peak-to-trough
- **Win Rate:** Profitable trade percentage
- **Profit Factor:** Gross profit / Gross loss

## Algorithm Details

### Graph Construction from Crypto Data

```
Input: OHLCV data for N cryptocurrencies
Parameters: window_size, correlation_threshold

1. Calculate returns: r_i = (close_t - close_{t-1}) / close_{t-1}
2. Compute correlation matrix: C = corr(returns, window=window_size)
3. Build adjacency:
   A[i,j] = 1 if C[i,j] > threshold
   A[i,j] = 0 otherwise
4. Create graph G = (V, E) where:
   V = {asset_1, ..., asset_N}
   E = {(i,j) : A[i,j] = 1}
```

### gSpan Core Algorithm

```
Procedure gSpan(G, min_sup):
    S = ∅  // frequent subgraphs
    for each frequent edge e in G:
        Pattern p = {e}
        gSpan-Extend(p, G, min_sup, S)
    return S

Procedure gSpan-Extend(p, G, min_sup, S):
    if p is closed:  // no frequent extensions
        S = S ∪ {p}
        return

    for each right-most extension e of p:
        p' = p ∪ {e}
        if support(p', G) >= min_sup:
            gSpan-Extend(p', G, min_sup, S)
```

### Trading Signal Logic

```
Signal Generation:
1. BULLISH patterns:
   - New triangle formation in uptrending assets
   - Star pattern with strong hub (market leader)
   - Clique expansion (more assets joining correlated group)

2. BEARISH patterns:
   - Triangle breaking (edge removal)
   - Star collapse (hub weakening)
   - Clique fragmentation

3. NEUTRAL / CAUTION:
   - Rapid pattern changes (regime uncertainty)
   - Isolated nodes increasing (decorrelation)
```

## Bybit Integration

### Data Sources
- **Spot Market:** BTCUSDT, ETHUSDT, and 50+ pairs
- **Perpetual Futures:** Funding rates, open interest
- **Order Book:** Depth snapshots (L2/L3)
- **Trades:** Individual transactions with timestamps

### API Endpoints Used
```
GET /v5/market/kline          # OHLCV candles
GET /v5/market/tickers        # Current prices
GET /v5/market/orderbook      # Order book depth
GET /v5/market/recent-trade   # Recent trades
```

## Expected Outcomes

1. **Graph Construction Pipeline:** Build correlation/flow graphs from Bybit data
2. **Mining Engine:** Efficient subgraph pattern discovery
3. **Pattern Catalog:** Library of significant trading patterns
4. **Signal Generator:** Real-time pattern matching and signals
5. **Backtest Framework:** Historical performance validation
6. **Live Trading Ready:** Production-quality Rust implementation

## References

### Academic Papers
- [gSpan: Graph-Based Substructure Pattern Mining](https://www.cs.ucsb.edu/~xyan/papers/gSpan-short.pdf) - Yan & Han, 2002
- [Subgraph Neural Networks](https://arxiv.org/abs/2006.10538) - Alsentzer et al., 2020
- [Network Motifs in Financial Markets](https://arxiv.org/abs/1209.2927) - Pozzi et al., 2012
- [Graph-based Stock Market Analysis](https://arxiv.org/abs/2004.00388) - Chen et al., 2020

### Technical Resources
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)
- [petgraph (Rust Graph Library)](https://docs.rs/petgraph/)
- [Graph Mining Survey](https://arxiv.org/abs/2005.03675)

## Difficulty Level

Expert

**Prerequisites:**
- Graph theory fundamentals
- Algorithm design (DFS, pattern matching)
- Time series analysis
- Rust programming
- Cryptocurrency market mechanics
