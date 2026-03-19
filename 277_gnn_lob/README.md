# Chapter 277: GNN LOB -- Graph Neural Network for Limit Order Book Trading

## 1. Introduction

The **Limit Order Book (LOB)** is the central data structure of modern electronic exchanges. It records all outstanding buy (bid) and sell (ask) orders at various price levels. Traditional approaches to LOB modeling -- such as feeding raw price-volume vectors into LSTMs or CNNs -- treat price levels as independent features arranged on a flat grid. This ignores the rich relational structure that naturally exists among price levels: neighboring levels interact through order flow, cancellation cascades, and liquidity migration.

**Graph Neural Networks (GNNs)** offer a principled way to model these relational dynamics. By representing each price level as a **node** in a graph, and connecting levels with **edges** that encode proximity, correlation, or causal influence, we can leverage message-passing mechanisms to learn representations that capture cross-level dependencies far more effectively than grid-based models.

### Why Graphs for LOB?

1. **Variable topology**: The number of active price levels changes tick by tick. Graphs handle variable-size inputs naturally, whereas fixed-size vectors require padding or truncation.
2. **Locality and long-range interactions**: A GCN with *k* layers captures *k*-hop neighborhoods. A two-layer network lets information flow from a bid level to an ask level two hops away, modeling cross-spread interactions.
3. **Permutation awareness**: While price levels have a natural ordering, the graph formulation lets us additionally encode non-sequential relationships (e.g., round-number effects at \$50,000 vs. \$50,100).
4. **Attention over levels**: Graph Attention Networks (GATs) learn which neighboring levels are most informative for predicting price movement, providing interpretable attention maps.

### Chapter Roadmap

| Section | Topic |
|---------|-------|
| 2 | Mathematical foundations -- graph construction, GCN, GAT |
| 3 | Applications -- mid-price prediction, cross-level dependencies, market impact |
| 4 | Rust implementation |
| 5 | Bybit data integration |
| 6 | Key takeaways |

---

## 2. Mathematical Foundations

### 2.1 Graph Construction from LOB

Given an LOB snapshot with *B* bid levels and *A* ask levels, we construct a graph **G = (V, E, X)** where:

- **Nodes** V = {v_1, ..., v_{B+A}}. Each node represents one price level.
- **Node features** X in R^{N x F}. For each level *i*, the feature vector is:

  x_i = [p_i, q_i, side_i, delta_p_i, delta_q_i, depth_ratio_i]

  where p_i is the price, q_i the quantity, side_i in {0, 1} indicates bid/ask, delta_p and delta_q are changes since the last snapshot, and depth_ratio is the cumulative depth fraction.

- **Edges** E. We consider several strategies:

  **Strategy 1 -- k-Nearest Price Levels:**

  Connect each node to its *k* closest neighbors by price:

  E_knn = {(i, j) : |p_i - p_j| <= delta_k(i)}

  where delta_k(i) is the distance to the k-th nearest neighbor of node *i*.

  **Strategy 2 -- Full Bipartite (Bid-Ask):**

  Connect every bid node to every ask node to model cross-spread dynamics:

  E_bipartite = {(i, j) : side_i != side_j}

  **Strategy 3 -- Sequential + Skip Connections:**

  Connect each node to its immediate neighbors on the same side, plus skip connections every *s* levels:

  E_seq = {(i, i+1)} union {(i, i+s)}

- **Edge weights** (optional): w_{ij} = exp(-alpha |p_i - p_j|) for a decay parameter alpha > 0.

### 2.2 Graph Convolutional Network (GCN) Layer

The GCN layer from Kipf and Welling (2017) performs neighborhood aggregation:

H^{(l+1)} = sigma(D_hat^{-1/2} A_hat D_hat^{-1/2} H^{(l)} W^{(l)})

where:
- A_hat = A + I_N (adjacency with self-loops)
- D_hat_{ii} = sum_j A_hat_{ij} (degree matrix)
- W^{(l)} in R^{F_l x F_{l+1}} is the learnable weight matrix
- sigma is a non-linear activation (ReLU)

For LOB graphs, this means each price level aggregates information from connected levels, weighted by the inverse square root of their degrees.

**Simplified per-node update:**

h_i^{(l+1)} = sigma(W^{(l)} * (1/(d_i+1)) * (h_i^{(l)} + sum_{j in N(i)} (1/sqrt((d_i+1)(d_j+1))) * h_j^{(l)}))

### 2.3 Graph Attention Network (GAT) Layer

The GAT (Velickovic et al., 2018) replaces fixed normalization with learned attention coefficients:

**Step 1 -- Attention coefficients:**

e_{ij} = LeakyReLU(a^T [W h_i || W h_j])

where || denotes concatenation and a in R^{2F'} is a learnable attention vector.

**Step 2 -- Softmax normalization:**

alpha_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / sum_{k in N(i)} exp(e_{ik})

**Step 3 -- Weighted aggregation:**

h_i' = sigma(sum_{j in N(i)} alpha_{ij} W h_j)

**Multi-head attention** with *K* heads:

h_i' = ||_{k=1}^{K} sigma(sum_{j in N(i)} alpha_{ij}^k W^k h_j)

For LOB data, attention heads can specialize: one head may focus on same-side neighbors (capturing depth pressure), while another attends to cross-spread levels (capturing spread dynamics).

### 2.4 Graph-Level Readout

After *L* message-passing layers, we obtain node embeddings {h_i^{(L)}}. To make a graph-level prediction (e.g., mid-price direction), we need a **readout** function:

**Mean pooling:**

h_G = (1/|V|) sum_{i in V} h_i^{(L)}

**Max pooling:**

h_G = max_{i in V} h_i^{(L)} (element-wise)

**Attention pooling:**

h_G = sum_{i in V} softmax(MLP(h_i^{(L)})) * h_i^{(L)}

The graph embedding h_G is then fed to a prediction head (MLP) for classification or regression.

### 2.5 Mid-Price Direction Prediction

The prediction target is the future mid-price movement:

y = sign(m_{t+tau} - m_t)

where m_t = (p_best_bid + p_best_ask) / 2.

The full model:

y_hat = MLP(ReadOut(GAT_L(...GAT_1(X, A)...)))

Loss function (cross-entropy for 3-class: up/down/stationary):

L = -sum_{c in {up, down, stat}} y_c log(y_hat_c)

---

## 3. Applications

### 3.1 LOB State Prediction

The primary application is predicting the next LOB state or mid-price direction. The GNN captures:

- **Depth imbalance**: Asymmetry between bid and ask volumes at multiple levels, aggregated through message passing.
- **Liquidity clusters**: Groups of levels with unusually high volume that act as support/resistance.
- **Queue position effects**: How the shape of the order book at different depths signals future price moves.

Empirical results from the literature (Zhang et al., 2019; Xu et al., 2021) show that GNN-based LOB models outperform DeepLOB and other CNN/LSTM baselines by 2-5% in F1 score on mid-price direction prediction.

### 3.2 Cross-Level Dependency Modeling

Traditional models process each level independently or through convolution over sequential levels. GNNs can model:

- **Bid-ask coupling**: How changes at the best ask affect deeper bid levels (and vice versa).
- **Cascading cancellations**: When a large order is placed at level *k*, it may trigger cancellations at levels *k+1*, *k+2*, etc. The graph propagates this information.
- **Hidden liquidity detection**: Iceberg orders create anomalous patterns in the local graph neighborhood that GNNs can detect.

### 3.3 Market Impact Modeling

For execution algorithms, understanding how a trade at one price level impacts the entire book is critical. The GNN framework naturally models this:

1. Encode the current LOB as a graph.
2. Simulate the removal of volume at a target level (the trade).
3. Use the GNN to predict the new equilibrium state of the LOB.

This enables optimal execution strategies that minimize market impact by choosing levels and timing based on the graph structure.

### 3.4 Multi-Asset LOB Graphs

The framework extends to multiple assets by creating a **multi-graph**:

- Each asset has its own LOB sub-graph.
- Cross-asset edges connect correlated price levels (e.g., BTC best bid to ETH best bid).
- The GNN then captures both intra-book and cross-book dynamics, useful for pairs trading and statistical arbitrage.

---

## 4. Rust Implementation

Our Rust implementation in `rust/src/lib.rs` provides:

| Component | Description |
|-----------|-------------|
| `LOBGraph` | Builds a graph from bid/ask levels with configurable adjacency |
| `GCNLayer` | Standard GCN message passing with learnable weights |
| `GATLayer` | Attention-based aggregation with multi-head support |
| `GraphReadout` | Mean and max pooling for graph-level embeddings |
| `MidPricePredictor` | Full pipeline: graph construction + GNN + prediction head |
| `BybitClient` | Fetches real-time orderbook data from Bybit API |

Key design decisions:

- **`ndarray`** for matrix operations -- provides NumPy-like ergonomics in Rust.
- **No external ML framework** -- all GNN layers are implemented from scratch to demonstrate the math.
- **Async Bybit client** -- uses `reqwest` for non-blocking API calls.

### Building and Running

```bash
cd 277_gnn_lob/rust
cargo build
cargo test
cargo run --example trading_example
```

---

## 5. Bybit Data Integration

The implementation connects to Bybit's public REST API:

**Endpoint:** `GET https://api.bybit.com/v5/market/orderbook`

**Parameters:**
- `category=spot`
- `symbol=BTCUSDT`
- `limit=50` (up to 200 levels)

**Response structure:**
```json
{
  "result": {
    "b": [["price", "qty"], ...],
    "a": [["price", "qty"], ...],
    "ts": 1234567890,
    "u": 12345
  }
}
```

The `BybitClient` parses this into `PriceLevel` structs and feeds them to `LOBGraph::from_orderbook()`.

### Data Pipeline

1. **Fetch** orderbook snapshot (50 levels each side).
2. **Construct** graph with k-nearest neighbor edges (k=5 default).
3. **Compute** node features: price, quantity, side, normalized position.
4. **Run** GNN forward pass (2 GCN layers + readout + MLP).
5. **Output** predicted mid-price direction and confidence.

---

## 6. Key Takeaways

1. **LOBs have natural graph structure.** Price levels interact through order flow, and GNNs capture these interactions through message passing.

2. **GCN vs. GAT trade-off.** GCN is faster and simpler; GAT provides attention-based interpretability and can adaptively weight neighbors. For LOB data, GAT typically outperforms GCN by learning which levels matter most.

3. **Graph construction matters.** The choice of edge strategy (k-NN, bipartite, sequential) significantly impacts performance. k-NN with k=5 is a good default; adding cross-spread edges further improves prediction.

4. **Scalability.** LOB graphs are small (tens to hundreds of nodes), so GNN inference is fast -- well within the latency budget for HFT applications.

5. **Multi-asset extension.** The same framework naturally extends to multi-asset graphs for cross-market signal extraction.

6. **Rust for production.** Our implementation demonstrates that GNN inference can be done in pure Rust without Python dependencies, enabling deployment in low-latency trading systems.

---

## References

- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
- Velickovic, P., et al. (2018). Graph Attention Networks. ICLR.
- Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. IEEE Trans. Signal Processing.
- Xu, K., et al. (2021). Graph Neural Networks for Limit Order Book Modeling. NeurIPS Workshop on ML for Financial Markets.
- Cont, R. (2011). Statistical Modeling of High-Frequency Financial Data. IEEE Signal Processing Magazine.
