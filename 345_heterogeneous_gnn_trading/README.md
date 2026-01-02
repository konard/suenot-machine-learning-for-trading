# Chapter 345: Heterogeneous Graph Neural Networks for Trading

## Overview

Heterogeneous Graph Neural Networks (Heterogeneous GNNs or HGNNs) extend traditional GNNs to handle graphs with multiple types of nodes and edges. In financial markets, this is crucial because the market ecosystem consists of diverse entities (assets, exchanges, traders, news sources) connected through various relationship types (correlations, order flow, ownership, influence).

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Heterogeneous Graph Construction](#heterogeneous-graph-construction)
4. [Architecture Components](#architecture-components)
5. [Metapath-Based Learning](#metapath-based-learning)
6. [Application to Cryptocurrency Trading](#application-to-cryptocurrency-trading)
7. [Implementation Strategy](#implementation-strategy)
8. [Risk Management](#risk-management)
9. [Performance Metrics](#performance-metrics)
10. [References](#references)

---

## Introduction

Financial markets are inherently heterogeneous systems. Unlike homogeneous graphs where all nodes represent the same entity type, real markets consist of:

- **Multiple node types**: Assets, exchanges, wallets, traders, news sources
- **Multiple edge types**: Price correlation, order flow, funding rate arbitrage, whale transfers
- **Rich semantic relationships**: Each connection type carries different meaning

### Why Heterogeneous GNNs for Trading?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Market as a Heterogeneous Graph                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Node Types:                     Edge Types:                            │
│   ┌─────────┐                    ═══════════                            │
│   │  Asset  │ ──correlation──→   Asset-Asset: price correlation         │
│   └─────────┘                    Asset-Exchange: trading venue          │
│        │                         Wallet-Asset: holdings                  │
│    trades_on                     Trader-Asset: positions                 │
│        ↓                         News-Asset: mentions                    │
│   ┌──────────┐                                                          │
│   │ Exchange │                                                           │
│   └──────────┘                                                          │
│        ↑                                                                 │
│    connected                                                             │
│        │                                                                 │
│   ┌─────────┐    influences    ┌─────────┐                              │
│   │  Whale  │ ────────────────→│  Retail │                              │
│   │ Wallet  │                  │ Traders │                              │
│   └─────────┘                  └─────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Advantages over Homogeneous GNNs

| Aspect | Homogeneous GNN | Heterogeneous GNN |
|--------|-----------------|-------------------|
| Node types | Single type | Multiple types |
| Edge semantics | Uniform | Type-specific |
| Feature spaces | Shared | Type-specific |
| Relationship modeling | Limited | Rich semantic paths |
| Market representation | Simplified | Realistic |

## Theoretical Foundation

### Heterogeneous Graph Definition

A heterogeneous graph is defined as $G = (V, E, \tau, \phi)$ where:

- **V**: Set of nodes
- **E**: Set of edges
- **τ: V → T_V**: Node type mapping function
- **φ: E → T_E**: Edge type mapping function

With constraints: $|T_V| + |T_E| > 2$ (more than one type of node or edge)

### Graph Schema

The schema defines allowed node and edge types:

```
Schema S = (T_V, T_E, R)

Where R defines valid relations:
  R: T_V × T_E × T_V

Examples for trading:
  (Asset, correlation, Asset)
  (Asset, trades_on, Exchange)
  (Wallet, holds, Asset)
  (Trader, executes, Order)
  (News, mentions, Asset)
```

### Heterogeneous Message Passing

The core operation extends message passing to handle type-specific transformations:

$$h_v^{(l+1)} = \text{AGG}\left(\left\{\text{MSG}_{\phi(e)}(h_u^{(l)}, h_v^{(l)}, e) : (u, e, v) \in \mathcal{N}(v)\right\}\right)$$

Where:
- $\text{MSG}_{\phi(e)}$ is a type-specific message function
- Different edge types use different transformation weights
- Aggregation considers type distribution

### Type-Specific Transformations

Each node/edge type has its own projection:

```
For node type t ∈ T_V:
    W_t: ℝ^(d_t) → ℝ^d  (project to common space)

For edge type r ∈ T_E:
    M_r: ℝ^d × ℝ^d → ℝ^d  (relation-specific transform)
```

## Heterogeneous Graph Construction

### Node Type Definitions

```
┌────────────────────────────────────────────────────────────────┐
│                     Node Types in Trading                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ASSET Node                                                  │
│     Features: [price, volume, volatility, market_cap,          │
│                funding_rate, open_interest, returns_1h/4h/24h] │
│                                                                 │
│  2. EXCHANGE Node                                               │
│     Features: [total_volume, num_pairs, liquidity_score,       │
│                avg_spread, reliability_score]                   │
│                                                                 │
│  3. WALLET Node (Whale Tracking)                                │
│     Features: [balance, tx_count, avg_tx_size, age,            │
│                pnl_estimate, activity_score]                    │
│                                                                 │
│  4. MARKET_CONDITION Node                                       │
│     Features: [btc_dominance, total_market_cap, fear_greed,    │
│                funding_avg, long_short_ratio]                   │
│                                                                 │
│  5. TIMEFRAME Node                                              │
│     Features: [trend_1m, trend_5m, trend_1h, trend_4h,         │
│                volatility_regime, volume_profile]               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Edge Type Definitions

```
┌────────────────────────────────────────────────────────────────┐
│                     Edge Types in Trading                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Asset ←→ Asset Edges:                                          │
│  • correlation: Rolling price correlation                       │
│  • cointegration: Statistical cointegration score              │
│  • lead_lag: Granger causality relationship                    │
│  • sector: Same sector/category                                 │
│                                                                 │
│  Asset ←→ Exchange Edges:                                       │
│  • trades_on: Asset is listed on exchange                      │
│  • liquidity: Volume/depth on specific exchange                │
│  • spread: Bid-ask spread on exchange                          │
│                                                                 │
│  Wallet ←→ Asset Edges:                                         │
│  • holds: Wallet holds this asset                              │
│  • accumulating: Increasing position                           │
│  • distributing: Decreasing position                           │
│                                                                 │
│  Asset ←→ MarketCondition Edges:                                │
│  • sensitivity: How asset reacts to market conditions          │
│  • beta: Market beta coefficient                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Building the Graph from Market Data

```python
# Pseudocode for graph construction
def build_heterogeneous_graph(market_data):
    graph = HeterogeneousGraph()

    # Add Asset nodes
    for asset in market_data.assets:
        features = extract_asset_features(asset)
        graph.add_node(asset.symbol, type="ASSET", features=features)

    # Add Exchange nodes
    for exchange in market_data.exchanges:
        features = extract_exchange_features(exchange)
        graph.add_node(exchange.name, type="EXCHANGE", features=features)

    # Add correlation edges between assets
    corr_matrix = compute_rolling_correlation(market_data.prices)
    for i, j in pairs_above_threshold(corr_matrix, 0.5):
        graph.add_edge(
            assets[i], assets[j],
            type="correlation",
            weight=corr_matrix[i,j]
        )

    # Add trading venue edges
    for asset in assets:
        for exchange in asset.exchanges:
            graph.add_edge(
                asset.symbol, exchange,
                type="trades_on",
                features=get_venue_features(asset, exchange)
            )

    return graph
```

## Architecture Components

### 1. Heterogeneous Graph Transformer (HGT)

The HGT architecture uses type-specific attention:

```
┌─────────────────────────────────────────────────────────────────┐
│              Heterogeneous Graph Transformer Layer               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Node features H^(l), Graph structure G                  │
│                                                                  │
│  For each target node t with type τ(t):                         │
│                                                                  │
│    1. Type-Specific Projection:                                 │
│       K_s = W_K^(τ(s)) · h_s    (source key)                   │
│       V_s = W_V^(τ(s)) · h_s    (source value)                 │
│       Q_t = W_Q^(τ(t)) · h_t    (target query)                 │
│                                                                  │
│    2. Relation-Specific Attention:                              │
│       For edge type e = (τ(s), φ(e), τ(t)):                    │
│         α_st = softmax(Q_t · W_ATT^e · K_s / √d)               │
│                                                                  │
│    3. Message Aggregation:                                      │
│       m_t = Σ α_st · W_MSG^e · V_s                             │
│                                                                  │
│    4. Update:                                                   │
│       h_t^(l+1) = LayerNorm(h_t^(l) + FFN(m_t))               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Relation-Aware Aggregation

```rust
// Conceptual Rust structure
struct RelationAggregator {
    // Separate weights for each relation type
    relation_weights: HashMap<EdgeType, Matrix>,
    // Attention mechanism
    attention: MultiHeadAttention,
}

impl RelationAggregator {
    fn aggregate(&self, node: &Node, neighbors: &[Neighbor]) -> Vector {
        let mut messages_by_type: HashMap<EdgeType, Vec<Vector>> = HashMap::new();

        // Group messages by relation type
        for neighbor in neighbors {
            let transformed = self.relation_weights[&neighbor.edge_type]
                .dot(&neighbor.features);
            messages_by_type
                .entry(neighbor.edge_type)
                .or_default()
                .push(transformed);
        }

        // Aggregate within each type, then across types
        let type_embeddings: Vec<Vector> = messages_by_type
            .iter()
            .map(|(etype, msgs)| self.aggregate_type(etype, msgs))
            .collect();

        self.attention.aggregate(&type_embeddings)
    }
}
```

### 3. Metapath Encoder

Metapaths capture semantic relationships through typed paths:

```
Example Metapaths for Trading:

1. Asset-correlation-Asset-correlation-Asset (A-c-A-c-A)
   "Assets correlated through intermediary"

2. Asset-trades_on-Exchange-trades_on-Asset (A-E-A)
   "Assets on same exchange"

3. Wallet-holds-Asset-correlation-Asset (W-A-A)
   "Whale portfolio correlation structure"

4. Asset-sensitivity-MarketCondition-sensitivity-Asset (A-M-A)
   "Assets with similar market sensitivity"
```

### 4. Semantic Attention Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                   Semantic Attention                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Given metapath-based embeddings Z^Φ for each metapath Φ:       │
│                                                                  │
│  1. Compute metapath importance:                                 │
│     w_Φ = (1/|V|) Σ_v tanh(a^T · Z_v^Φ)                        │
│                                                                  │
│  2. Normalize across metapaths:                                  │
│     β_Φ = softmax(w_Φ)                                          │
│                                                                  │
│  3. Final embedding:                                             │
│     Z_final = Σ_Φ β_Φ · Z^Φ                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Metapath-Based Learning

### Key Metapaths for Crypto Trading

```
┌────────────────────────────────────────────────────────────────────┐
│                    Trading-Specific Metapaths                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Metapath 1: Cross-Asset Momentum (A-corr-A)                       │
│  ─────────────────────────────────────────────                     │
│  BTC ──correlation──→ ETH                                          │
│  Captures: Direct price co-movement                                │
│  Trading use: Pair trading, momentum spillover                     │
│                                                                     │
│  Metapath 2: Exchange Arbitrage (A-E-A)                            │
│  ─────────────────────────────────────────────                     │
│  BTC ──trades_on──→ Bybit ←──trades_on── BTC                       │
│                         ↓                                           │
│                      Binance                                        │
│  Captures: Cross-exchange price discrepancies                      │
│  Trading use: Arbitrage opportunity detection                      │
│                                                                     │
│  Metapath 3: Whale Following (W-holds-A-corr-A)                    │
│  ─────────────────────────────────────────────                     │
│  Whale ──holds──→ SOL ──correlation──→ AVAX                        │
│  Captures: Smart money flow patterns                               │
│  Trading use: Follow whale accumulation                            │
│                                                                     │
│  Metapath 4: Market Regime (A-sens-M-sens-A)                       │
│  ─────────────────────────────────────────────                     │
│  ETH ──sensitivity──→ BullMarket ←──sensitivity── SOL              │
│  Captures: Assets with similar regime behavior                     │
│  Trading use: Regime-based portfolio construction                  │
│                                                                     │
│  Metapath 5: Sector Rotation (A-sector-A-sector-A)                 │
│  ─────────────────────────────────────────────                     │
│  AAVE ──DeFi──→ UNI ──DeFi──→ COMP                                │
│  Captures: Sector-wide movements                                   │
│  Trading use: Sector momentum strategies                           │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Metapath Instance Sampling

```
Algorithm: Random Walk with Restart for Metapath Sampling

Input: Graph G, metapath schema Φ, start node v, num_walks n

For i = 1 to n:
    path = [v]
    current = v
    for step in Φ.edge_types:
        # Get valid neighbors for this edge type
        valid_neighbors = get_neighbors(current, step.edge_type)
        if empty(valid_neighbors):
            break
        # Sample next node
        next_node = sample(valid_neighbors, weights=edge_weights)
        path.append(next_node)
        current = next_node

    yield path
```

## Application to Cryptocurrency Trading

### Bybit Market Heterogeneous Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Bybit Heterogeneous Trading Graph                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      ASSET LAYER                             │   │
│  │  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐       │   │
│  │  │ BTC │════│ ETH │════│ SOL │════│ AVAX│════│ ARB │       │   │
│  │  └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘       │   │
│  │     │          │          │          │          │           │   │
│  │     │ correlation edges   │ lead-lag edges     │           │   │
│  └─────│──────────│──────────│──────────│──────────│───────────┘   │
│        │          │          │          │          │                │
│  ┌─────│──────────│──────────│──────────│──────────│───────────┐   │
│  │     │   MARKET STRUCTURE LAYER       │          │           │   │
│  │  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐        │   │
│  │  │ Spot │  │ Perp │  │ Spot │  │ Perp │  │ Perp │        │   │
│  │  │ BTC  │  │ BTC  │  │ ETH  │  │ ETH  │  │ SOL  │        │   │
│  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘        │   │
│  │     │ funding  │ basis   │          │          │           │   │
│  │     └────┬─────┘         └────┬─────┘          │           │   │
│  └──────────│────────────────────│────────────────│────────────┘   │
│             │                    │                │                 │
│  ┌──────────│────────────────────│────────────────│────────────┐   │
│  │          │   WHALE LAYER      │                │            │   │
│  │  ┌───────┴──────┐  ┌─────────┴────────┐  ┌───┴─────┐      │   │
│  │  │ Whale Wallet │  │ Whale Wallet     │  │ Retail  │      │   │
│  │  │     #1       │  │     #2           │  │  Pool   │      │   │
│  │  └──────────────┘  └──────────────────┘  └─────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering by Node Type

| Node Type | Features | Dimension | Update Freq |
|-----------|----------|-----------|-------------|
| Asset | price, volume, returns, volatility, OI, funding | 32 | Real-time |
| SpotMarket | bid, ask, spread, depth, imbalance | 16 | Real-time |
| PerpMarket | mark_price, index_price, funding_rate, OI | 24 | Real-time |
| Wallet | balance, pnl, tx_count, avg_size | 12 | Minutes |
| MarketRegime | trend, volatility_regime, correlation_regime | 8 | Hourly |

### Trading Signal Generation Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│              Heterogeneous GNN Signal Pipeline                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Data Ingestion                                                  │
│     ├── Bybit REST API: Klines, Tickers, Order Book               │
│     ├── Bybit WebSocket: Real-time trades, depth                  │
│     └── On-chain data: Whale movements (optional)                  │
│                                                                     │
│  2. Graph Construction                                              │
│     ├── Create/Update nodes by type                                │
│     ├── Compute correlation edges (rolling window)                 │
│     ├── Update market structure edges                              │
│     └── Refresh whale position edges                               │
│                                                                     │
│  3. HGNN Forward Pass                                               │
│     ├── Type-specific feature projection                           │
│     ├── Relation-aware message passing                             │
│     ├── Metapath attention aggregation                             │
│     └── Generate node embeddings                                   │
│                                                                     │
│  4. Prediction Heads                                                │
│     ├── Direction: Classify up/down/neutral                        │
│     ├── Magnitude: Regress expected return                         │
│     ├── Confidence: Estimate prediction uncertainty                │
│     └── Edge prediction: New relationship emergence                │
│                                                                     │
│  5. Signal Aggregation                                              │
│     ├── Combine predictions across metapaths                       │
│     ├── Weight by historical accuracy                              │
│     └── Generate final trading signal                              │
│                                                                     │
│  6. Position Sizing & Execution                                     │
│     ├── Kelly criterion with type-specific risk                    │
│     ├── Cross-asset correlation adjustment                         │
│     └── Execute via Bybit API                                      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Module Architecture

```
heterogeneous_gnn_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Library root
│   ├── graph/
│   │   ├── mod.rs               # Graph module
│   │   ├── node.rs              # Typed node definitions
│   │   ├── edge.rs              # Typed edge definitions
│   │   ├── schema.rs            # Graph schema
│   │   └── heterogeneous.rs     # Heterogeneous graph ops
│   ├── gnn/
│   │   ├── mod.rs               # GNN module
│   │   ├── layers.rs            # HGNN layers
│   │   ├── attention.rs         # Type-aware attention
│   │   ├── metapath.rs          # Metapath encoder
│   │   └── aggregation.rs       # Semantic aggregation
│   ├── data/
│   │   ├── mod.rs               # Data module
│   │   ├── bybit.rs             # Bybit API client
│   │   ├── features.rs          # Feature engineering
│   │   └── types.rs             # Market data types
│   ├── strategy/
│   │   ├── mod.rs               # Strategy module
│   │   ├── signals.rs           # Signal generation
│   │   └── execution.rs         # Order execution
│   └── utils/
│       ├── mod.rs               # Utilities
│       └── metrics.rs           # Performance metrics
├── examples/
│   ├── basic_hgnn.rs            # Basic HGNN example
│   ├── live_trading.rs          # Live trading demo
│   └── backtest.rs              # Backtesting
└── tests/
    └── integration.rs           # Integration tests
```

### Key Design Principles

1. **Type Safety**: Leverage Rust's type system for node/edge types
2. **Modularity**: Each component is independent and testable
3. **Performance**: Efficient sparse operations for large graphs
4. **Extensibility**: Easy to add new node/edge types
5. **Real-time Ready**: Designed for streaming updates

### Example: Defining Node Types in Rust

```rust
/// Node type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Asset,
    Exchange,
    Wallet,
    MarketCondition,
    Timeframe,
}

/// Edge type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    // Asset-Asset
    Correlation,
    Cointegration,
    LeadLag,
    SameSector,
    // Asset-Exchange
    TradesOn,
    // Wallet-Asset
    Holds,
    Accumulating,
    Distributing,
    // Asset-MarketCondition
    Sensitivity,
}

/// Graph schema defining valid relations
pub struct GraphSchema {
    valid_relations: HashSet<(NodeType, EdgeType, NodeType)>,
}

impl GraphSchema {
    pub fn trading_schema() -> Self {
        let mut relations = HashSet::new();

        // Asset-Asset relations
        relations.insert((NodeType::Asset, EdgeType::Correlation, NodeType::Asset));
        relations.insert((NodeType::Asset, EdgeType::LeadLag, NodeType::Asset));
        relations.insert((NodeType::Asset, EdgeType::SameSector, NodeType::Asset));

        // Asset-Exchange relations
        relations.insert((NodeType::Asset, EdgeType::TradesOn, NodeType::Exchange));

        // Wallet-Asset relations
        relations.insert((NodeType::Wallet, EdgeType::Holds, NodeType::Asset));
        relations.insert((NodeType::Wallet, EdgeType::Accumulating, NodeType::Asset));

        Self { valid_relations: relations }
    }
}
```

## Risk Management

### Type-Aware Risk Metrics

```
┌────────────────────────────────────────────────────────────────────┐
│                   Risk by Node/Edge Type                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Asset Risk Factors:                                                │
│  • Volatility risk (historical vol)                                │
│  • Liquidity risk (order book depth)                               │
│  • Funding rate risk (for perpetuals)                              │
│  • Correlation breakdown risk                                       │
│                                                                     │
│  Relationship Risk Factors:                                         │
│  • Correlation instability (rolling correlation variance)          │
│  • Lead-lag reversal probability                                   │
│  • Exchange-specific risks (withdrawal, API)                       │
│  • Whale behavior uncertainty                                       │
│                                                                     │
│  Graph Structure Risk:                                              │
│  • Clustering coefficient changes                                   │
│  • Centrality shifts                                               │
│  • New edge emergence rate                                         │
│  • Community structure stability                                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Position Limits by Type

```
┌────────────────────────────────────────────────────────────────┐
│                    Position Constraints                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Per-Asset Limits:                                              │
│  • Max position: 5% of portfolio                               │
│  • Max leverage: 3x                                            │
│  • Min liquidity: $1M daily volume                             │
│                                                                 │
│  Cross-Asset Limits:                                            │
│  • Max correlated exposure: 15%                                │
│  • Max sector exposure: 20%                                    │
│  • Max same-exchange exposure: 40%                             │
│                                                                 │
│  Type-Specific Adjustments:                                     │
│  • High-correlation pairs: reduce by correlation factor        │
│  • Whale-influenced assets: additional uncertainty buffer      │
│  • Low-liquidity venues: wider stop-loss                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Circuit Breakers

1. **Correlation Regime Change**: Pause if average cross-asset correlation shifts >2σ
2. **Type Distribution Shift**: Alert if node type proportions change significantly
3. **Metapath Breakdown**: Reduce exposure if key metapaths lose predictive power
4. **Whale Activity Spike**: Increase caution during unusual whale movements

## Performance Metrics

### Model Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| Node Classification Accuracy | Predict node type correctly | > 95% |
| Edge Type Prediction | Classify edge relationships | > 85% |
| Link Prediction AUC | Predict new edges | > 0.80 |
| Direction Prediction | Price movement direction | > 55% |
| Metapath Importance | Rank metapath predictive power | Stable |

### Trading Performance

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted returns | > 2.0 |
| Sortino Ratio | Downside risk-adjusted | > 2.5 |
| Max Drawdown | Largest peak-to-trough | < 15% |
| Win Rate | Profitable trades | > 52% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |

### Latency Budget

```
┌─────────────────────────────────────────────────┐
│              Latency Requirements               │
├─────────────────────────────────────────────────┤
│ Data Ingestion:          < 10ms                 │
│ Graph Update:            < 30ms                 │
│ Type Projection:         < 20ms                 │
│ Message Passing (3 hop): < 80ms                 │
│ Metapath Aggregation:    < 40ms                 │
│ Signal Generation:       < 20ms                 │
├─────────────────────────────────────────────────┤
│ Total Round Trip:        < 200ms                │
└─────────────────────────────────────────────────┘
```

## References

1. **Heterogeneous Graph Transformer**
   - Hu, Z., et al. (2020). "Heterogeneous Graph Transformer." *WWW 2020*
   - URL: https://arxiv.org/abs/2003.01332

2. **Heterogeneous Graph Attention Network (HAN)**
   - Wang, X., et al. (2019). "Heterogeneous Graph Attention Network." *WWW 2019*

3. **Metapath2Vec**
   - Dong, Y., et al. (2017). "metapath2vec: Scalable Representation Learning for Heterogeneous Networks." *KDD 2017*

4. **Relational Graph Convolutional Networks**
   - Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." *ESWC 2018*

5. **Financial Heterogeneous Networks**
   - Cheng, D., et al. (2021). "Financial Event Prediction with Heterogeneous Information." *ACM TKDD*

6. **Crypto Market Structure**
   - Makarov, I., & Schoar, A. (2020). "Trading and Arbitrage in Cryptocurrency Markets." *Journal of Financial Economics*

---

## Next Steps

- [View Simple Explanation](readme.simple.md) - Beginner-friendly version
- [Russian Version](README.ru.md) - Русская версия
- [Run Examples](examples/) - Working Rust code

---

*Chapter 345 of Machine Learning for Trading*
