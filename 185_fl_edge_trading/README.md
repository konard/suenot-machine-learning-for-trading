# Chapter 185: Edge Federated Learning for Trading

## 1. Introduction: Edge Computing Meets Federated Learning for Low-Latency Trading

In modern quantitative trading, latency is a first-class concern. Every microsecond matters when executing strategies that depend on rapid price movements, order book changes, and cross-exchange arbitrage. Traditional centralized machine learning pipelines introduce unavoidable round-trip delays: raw market data must travel from exchange colocations to a central data center, be processed, fed through a model, and the resulting signal must travel back. This architecture is fundamentally at odds with the speed requirements of high-frequency and medium-frequency trading.

Edge computing offers a compelling alternative. By deploying lightweight inference and training nodes directly at exchange colocations, within broker infrastructure, or even on mobile trading devices, we can dramatically reduce the latency between observation and action. However, edge deployment introduces its own challenges: limited compute resources, heterogeneous hardware, intermittent connectivity, and the risk of model fragmentation across isolated nodes.

Federated Learning (FL) bridges this gap. Rather than centralizing all data for training, FL allows each edge device to train locally on its own data stream and periodically share model updates with a coordinator. The coordinator aggregates these updates to produce a global model that benefits from the collective intelligence of all edge nodes, without ever centralizing the raw data.

This chapter explores the intersection of edge computing and federated learning for trading applications. We will design an architecture where edge devices at exchange colocations learn from local order flow, share compressed model updates, and collectively build a robust trading signal model. We will use FedProx to handle the inherent heterogeneity of edge devices and implement gradient compression to minimize communication overhead.

## 2. Edge Architecture for Trading

### 2.1 Edge Devices at Exchange Colocations

The primary deployment target for edge FL in trading is the colocation facility. Major exchanges like Binance, Bybit, CME, and NYSE offer colocation services where traders can place their servers physically close to the exchange matching engine. In our architecture, each colocation node runs:

- **A local data ingestion pipeline** that captures real-time order book snapshots, trade ticks, and funding rate updates directly from the exchange feed.
- **A lightweight ML model** (typically a small neural network or gradient-boosted ensemble) that generates trading signals from the local data.
- **A federated learning client** that periodically trains the model on recent local data and produces model updates for the coordinator.

The key insight is that each colocation node sees a slightly different view of the market. A node at Bybit sees Bybit-specific order flow patterns, while a node at Binance sees Binance-specific patterns. By federating across these nodes, the global model captures cross-exchange dynamics that no single node could learn alone.

### 2.2 Mobile and Remote Trading Nodes

Beyond colocations, edge FL extends to mobile trading applications and remote trading desks. These devices have significantly less compute power and more intermittent connectivity, but they contribute valuable signal diversity. A mobile trader in Asia may observe different market microstructure patterns than a colocation node in New York.

### 2.3 Hierarchical Edge Topology

We adopt a two-tier architecture:

1. **Tier 1 (Colocation Edges):** High-performance nodes with GPU acceleration, low-latency exchange feeds, and reliable connectivity. These nodes perform full model training and frequent updates.
2. **Tier 2 (Mobile/Remote Edges):** Resource-constrained nodes that perform partial model updates and communicate less frequently.

The coordinator sits in a cloud data center and orchestrates aggregation across both tiers, weighting contributions by device capability and data quality.

## 3. Mathematical Foundation

### 3.1 FedProx for Heterogeneous Edge Devices

Standard FedAvg assumes that all participating devices perform the same amount of local computation (e.g., the same number of local epochs). This assumption breaks down at the edge, where devices have vastly different compute budgets. A colocation server with a GPU can run 50 local epochs in the time a mobile device completes 2.

FedProx (Federated Proximal) addresses this by adding a proximal term to the local objective. For device $k$ at communication round $t$, the local objective becomes:

$$h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2$$

where:
- $F_k(w)$ is the local empirical loss on device $k$'s data
- $w^t$ is the current global model parameters
- $\mu \geq 0$ is the proximal term coefficient
- $\|w - w^t\|^2$ penalizes local models that drift too far from the global model

The proximal term has two critical effects:
1. **Stabilizes training on heterogeneous devices:** Devices that perform fewer local steps naturally stay closer to $w^t$, while devices that perform many steps are regularized toward $w^t$.
2. **Improves convergence:** The proximal term ensures bounded dissimilarity between local and global objectives.

### 3.2 Partial Model Updates

Not all edge devices can afford to update the entire model. We support partial updates where device $k$ only updates a subset $S_k$ of model parameters:

$$w_{k,i}^{t+1} = \begin{cases} w_{k,i}^{t} - \eta \nabla_{w_i} h_k(w_k^t; w^t) & \text{if } i \in S_k \\ w_i^t & \text{if } i \notin S_k \end{cases}$$

The coordinator tracks which parameters each device updated and performs weighted averaging only over the parameters that were actually trained.

### 3.3 Convergence Guarantee

Under standard assumptions (L-Lipschitz smooth, bounded variance $\sigma^2$, bounded dissimilarity $B$), FedProx with partial updates converges at rate:

$$\mathbb{E}[F(w^T)] - F^* \leq \mathcal{O}\left(\frac{1}{\sqrt{T}} + \frac{B + \sigma^2}{\mu T}\right)$$

where $T$ is the number of communication rounds. The proximal term $\mu$ provides a knob to trade off convergence speed against local computation flexibility.

## 4. Communication Efficiency: Gradient Compression and Quantization

### 4.1 The Communication Bottleneck

In edge FL for trading, communication bandwidth is precious. Colocation nodes may share network infrastructure with latency-sensitive trading systems, and mobile nodes operate over cellular networks. We cannot afford to transmit full model updates at every round.

### 4.2 Top-K Gradient Sparsification

We transmit only the top-K largest gradient components by magnitude:

$$\text{TopK}(\nabla F, K) = \{(\nabla F)_i : |(\nabla F)_i| \text{ is among the K largest}\}$$

Typically, K is set to 1-10% of the total parameter count. The remaining gradient components are accumulated in a local residual buffer and added to the next round's gradient, ensuring no information is permanently lost.

### 4.3 Stochastic Quantization

Each transmitted gradient component is quantized to reduce bit-width:

$$Q(v) = \|v\| \cdot \text{sign}(v_i) \cdot \xi_i(v, s)$$

where $\xi_i(v, s)$ is a stochastic rounding function with $s$ quantization levels. With $s = 256$ (8-bit quantization), we reduce communication by 4x compared to 32-bit floats while maintaining near-lossless model quality.

### 4.4 Combined Compression Ratio

Using Top-5% sparsification with 8-bit quantization, the total compression ratio is:

$$\text{Compression} = \frac{1}{0.05} \times \frac{32}{8} = 80\times$$

This means a model with 1 million parameters (4 MB in float32) requires only about 50 KB per update, well within the bandwidth constraints of even mobile networks.

## 5. Implementation Walkthrough

Our Rust implementation consists of several core components:

### 5.1 Edge Device Simulation

Each `EdgeDevice` struct represents a node with:
- **Compute capability:** Determines how many local training epochs the device can perform per round.
- **Local data buffer:** Stores recent market data for local training.
- **Model parameters:** A local copy of the model that gets updated through training.
- **Residual buffer:** Accumulates gradient components that were not transmitted due to Top-K sparsification.

### 5.2 FedProx Training Loop

The local training loop implements the proximal objective. At each local epoch:
1. Compute the standard loss gradient on a mini-batch of local data.
2. Add the proximal gradient: $\mu (w - w^t)$.
3. Update parameters via SGD.
4. If the device's compute budget is exhausted, stop early (partial work is valid under FedProx).

### 5.3 Gradient Compression Pipeline

After local training completes:
1. Compute the update delta: $\Delta w = w_{\text{local}} - w^t$.
2. Add the residual from previous rounds.
3. Apply Top-K sparsification to select the largest components.
4. Quantize selected components to 8-bit representation.
5. Store the unselected components in the residual buffer.
6. Transmit the compressed update to the coordinator.

### 5.4 Coordinator Aggregation

The coordinator receives compressed updates from participating devices and:
1. Decompresses and dequantizes each update.
2. Computes a weighted average based on device data sizes and compute contributions.
3. Applies the aggregated update to the global model.
4. Broadcasts the new global model to all devices.

### 5.5 Bybit Data Integration

The implementation includes a Bybit API client that fetches real BTCUSDT kline (candlestick) data. Each edge device processes this data into features (returns, volatility, momentum indicators) and targets (next-period direction or return). This provides realistic training data for the federated learning simulation.

## 6. Bybit Data Integration

### 6.1 API Endpoints

We use the Bybit public market data API:
- **Klines:** `GET /v5/market/kline?category=linear&symbol=BTCUSDT&interval=5`
- **Order Book:** `GET /v5/market/orderbook?category=linear&symbol=BTCUSDT`

### 6.2 Feature Engineering

From raw kline data, each edge device computes:

| Feature | Formula | Description |
|---------|---------|-------------|
| Return | $(close_t - close_{t-1}) / close_{t-1}$ | Price return |
| Volatility | $\text{std}(returns_{t-n:t})$ | Rolling standard deviation of returns |
| Momentum | $(close_t - close_{t-n}) / close_{t-n}$ | N-period momentum |
| Volume Ratio | $volume_t / \text{mean}(volume_{t-n:t})$ | Relative volume |
| Range | $(high_t - low_t) / close_t$ | Normalized candle range |

### 6.3 Data Distribution Across Edges

To simulate realistic data heterogeneity, each edge device receives a different time slice or subsample of the data. This creates non-IID data distributions that reflect the real-world scenario where different edge nodes observe different market conditions.

## 7. Key Takeaways

1. **Edge FL reduces latency:** By training and inferring at the edge, we eliminate the round-trip to a central data center. Models can react to market events in microseconds rather than milliseconds.

2. **FedProx handles heterogeneity gracefully:** The proximal term allows devices with vastly different compute capabilities to contribute meaningfully. A mobile device performing 2 local epochs contributes alongside a colocation GPU performing 50 epochs.

3. **Gradient compression is essential at the edge:** With 80x compression through Top-K sparsification and quantization, model updates fit within tight bandwidth constraints without meaningful loss in model quality.

4. **Partial updates enable resource-constrained participation:** Devices that cannot afford to update the entire model can update a subset of parameters. The coordinator handles the bookkeeping.

5. **Cross-exchange federation captures unique signals:** By federating across nodes at different exchanges, the global model learns cross-exchange dynamics (e.g., price lead-lag relationships, liquidity patterns) that no single node could discover.

6. **Privacy and regulatory compliance:** Raw trading data never leaves the edge device, which can simplify compliance with data residency regulations and protect proprietary trading strategies.

7. **Communication rounds should be adaptive:** In volatile markets, more frequent aggregation captures rapid regime changes. In calm markets, less frequent aggregation saves bandwidth.

8. **The Rust implementation demonstrates production-readiness:** Using Rust for the edge FL runtime provides the performance guarantees (no GC pauses, predictable latency) required for trading systems, while the type system catches many errors at compile time.

---

**Next Chapter Preview:** In Chapter 186, we will explore Federated Learning with Differential Privacy for Trading, adding formal privacy guarantees to the federated training process to protect sensitive trading strategies from inference attacks.
