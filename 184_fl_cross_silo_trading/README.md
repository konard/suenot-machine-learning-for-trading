# Chapter 184: Cross-Silo Federated Learning for Trading

## 1. Introduction

Cross-silo federated learning (FL) represents a paradigm shift in how institutional trading firms can collaborate on model training without surrendering their proprietary data. Unlike cross-device FL, which coordinates millions of small edge devices, cross-silo FL involves a smaller number of powerful institutional participants -- hedge funds, banks, proprietary trading firms, and asset managers -- each holding large, curated datasets of market signals, execution logs, and alpha factors.

The core promise is simple yet transformative: multiple trading institutions can jointly train a predictive model that is superior to any model trained on a single institution's data, while ensuring that no participant ever sees another's raw data. This is particularly compelling in financial markets where data is the competitive moat -- firms will never share order flow data, proprietary features, or execution statistics in the clear.

In traditional centralized machine learning, all data must be pooled into a single location. For trading firms, this is a non-starter due to regulatory constraints (MiFID II, GDPR, Dodd-Frank), competitive concerns, and the sheer sensitivity of trading data. Cross-silo FL eliminates this barrier by keeping data local and only exchanging model updates.

This chapter covers the architecture, mathematics, privacy guarantees, and a full Rust implementation of a cross-silo FL system designed for institutional trading, with integration into Bybit market data.

## 2. Architecture

### 2.1 Horizontal vs. Vertical Federated Learning

**Horizontal FL** applies when institutions share the same feature space but hold data on different samples. For example, three hedge funds each trading the same universe of crypto assets on Bybit, but each with different time windows, execution data, or proprietary signal overlays. The model architecture is identical across silos; what differs is the data distribution.

**Vertical FL** applies when institutions hold different features for overlapping samples. For instance, one firm has order book microstructure features, another has on-chain analytics, and a third has sentiment data -- all for the same set of assets over the same period. Vertical FL requires more sophisticated protocols (split learning, entity alignment) and is more complex to implement securely.

This chapter focuses primarily on horizontal FL, as it is the more common and practical starting point for trading applications.

### 2.2 Topology Options

**Star (Hub-and-Spoke) Topology**: A central aggregation server coordinates training rounds. Each institutional client trains locally and sends model updates to the server, which aggregates them and distributes the global model back. This is the simplest architecture but introduces a single point of failure and a trust assumption on the server.

**Ring Topology**: Model updates pass sequentially from one institution to the next, with each adding its contribution before forwarding. This eliminates the central server but increases latency linearly with the number of participants and makes the system vulnerable to any single node failure.

**Peer-to-Peer (Decentralized) Topology**: Each institution communicates directly with a subset of peers. Updates are gossiped through the network and converge through iterative averaging. This is the most resilient topology but the hardest to implement efficiently, especially with secure aggregation.

Our implementation uses the **star topology** with secure aggregation, which provides the best balance of simplicity, efficiency, and privacy for a typical cross-silo trading scenario with 3-10 institutional participants.

## 3. Mathematical Foundations

### 3.1 Weighted Federated Averaging (FedAvg)

The cornerstone algorithm for cross-silo FL is Federated Averaging. Given K institutional participants, each with a local dataset D_k of size n_k, the global model parameters at round t+1 are computed as:

```
w^{t+1} = sum_{k=1}^{K} (n_k / n) * w_k^{t+1}
```

where `n = sum_{k=1}^{K} n_k` is the total number of samples across all silos, and `w_k^{t+1}` are the locally updated parameters from participant k after E epochs of local SGD starting from `w^t`.

Each participant k performs local training:

```
w_k^{t+1} = w_k^t - eta * (1/n_k) * sum_{i in D_k} gradient(L(w_k^t; x_i, y_i))
```

The weighting by `n_k / n` ensures that institutions with more data have proportionally more influence on the global model, which is statistically optimal under the assumption of IID data across silos.

### 3.2 Handling Non-IID Data

In practice, trading data across institutions is rarely IID. Each firm has different trading strategies, asset coverage, and market access. To address this, several modifications to FedAvg are used:

- **FedProx**: Adds a proximal term `(mu/2) * ||w - w^t||^2` to each local objective, preventing local models from diverging too far from the global model.
- **SCAFFOLD**: Uses control variates to correct for client drift, converging faster on non-IID data.
- **Per-participant learning rates**: Allow each institution to adapt its local learning rate based on its data characteristics.

### 3.3 Secure Aggregation Protocol

Secure aggregation ensures that the central server can compute the weighted average of model updates without learning any individual participant's update. The protocol works as follows:

1. **Setup Phase**: Each pair of participants (i, j) agrees on a shared random seed s_{ij} using Diffie-Hellman key exchange.

2. **Masking Phase**: Participant k computes a masked update:
   ```
   y_k = w_k + sum_{j > k} PRG(s_{kj}) - sum_{j < k} PRG(s_{jk})
   ```
   where PRG is a pseudorandom generator.

3. **Aggregation Phase**: The server sums all masked updates:
   ```
   sum_{k} y_k = sum_{k} w_k
   ```
   The random masks cancel out in the sum, yielding the true aggregate while hiding individual contributions.

4. **Dropout Handling**: If a participant drops out, surviving participants can reconstruct the dropped party's mask using secret sharing (Shamir's scheme) to ensure the protocol completes.

## 4. Privacy Mechanisms

### 4.1 Differential Privacy

Even with secure aggregation, the global model update can leak information about individual participants' data through model inversion or membership inference attacks. Differential privacy (DP) provides a formal guarantee by adding calibrated noise.

**(epsilon, delta)-differential privacy** guarantees that for any two adjacent datasets D and D' (differing by one record), and any set of outputs S:

```
Pr[M(D) in S] <= exp(epsilon) * Pr[M(D') in S] + delta
```

In federated learning, we apply DP at two levels:

- **Local DP (LDP)**: Each participant clips their gradient to norm C and adds Gaussian noise N(0, sigma^2 * C^2 * I) before sending updates. This provides participant-level privacy but can significantly degrade model quality.

- **Central DP (CDP)**: The server adds noise to the aggregated update. Combined with secure aggregation, this provides record-level privacy with much less noise than LDP, as the noise is added once rather than K times.

For trading models, we recommend CDP with secure aggregation, targeting epsilon in the range of 1-10, which provides meaningful privacy with acceptable utility loss.

### 4.2 Secure Multi-Party Computation (MPC)

MPC generalizes secure aggregation to arbitrary computations. In the context of cross-silo FL for trading, MPC enables:

- **Private set intersection**: Determining which assets are traded by multiple participants without revealing full portfolios.
- **Secure comparison**: Comparing model performance across institutions without revealing actual metrics.
- **Oblivious transfer**: Allowing participants to selectively share features for vertical FL without revealing which features were selected.

MPC protocols are computationally expensive but feasible in the cross-silo setting where communication rounds are limited (typically 50-200 FL rounds) and bandwidth between institutional data centers is high.

### 4.3 Threat Model

Our system considers:

- **Honest-but-curious (semi-honest) participants**: They follow the protocol but attempt to infer information from observed messages. Secure aggregation defends against this.
- **Malicious server**: The aggregation server might attempt to reconstruct individual updates. Secure aggregation and DP defend against this.
- **Collusion**: Up to t < K/2 participants may collude with the server. Threshold secret sharing in the secure aggregation protocol defends against this.

We do not consider fully malicious participants who send corrupted updates; defending against Byzantine adversaries requires additional techniques (robust aggregation rules such as Krum or trimmed mean) that are discussed briefly in the implementation.

## 5. Implementation Walkthrough

The Rust implementation in this chapter provides a complete cross-silo FL system with the following components:

### 5.1 Core Architecture

- **`FederatedCoordinator`**: The central server that orchestrates training rounds, performs weighted aggregation, and manages participant registration. It maintains the global model and coordinates the secure aggregation protocol.

- **`InstitutionalClient`**: Represents a single institutional participant (e.g., a hedge fund). Each client holds its local dataset, performs local training (gradient descent on a linear model), and applies privacy mechanisms (gradient clipping and noise injection) before submitting updates.

- **`SecureAggregator`**: Simulates the secure aggregation protocol. In a production system, this would involve real cryptographic operations; here, we demonstrate the masking and unmasking logic with pseudorandom generators seeded by pairwise shared secrets.

- **`DifferentialPrivacy`**: Implements gradient clipping and Gaussian noise injection for local differential privacy.

### 5.2 Training Loop

The federated training proceeds in rounds:

1. The coordinator broadcasts the current global model to all participants.
2. Each participant trains locally for E epochs on its private dataset.
3. Each participant clips gradients and adds DP noise.
4. Participants submit masked updates through secure aggregation.
5. The coordinator aggregates updates using weighted FedAvg.
6. Repeat until convergence or budget exhaustion.

### 5.3 Model

We use a simple linear regression model (weights and bias) for price prediction. While real institutional models are far more complex, the linear model clearly demonstrates the FL mechanics without obscuring them in neural network complexity. The same FL infrastructure works with any differentiable model.

## 6. Bybit Data Integration

The implementation includes a Bybit data fetcher that retrieves recent kline (candlestick) data for specified trading pairs via the Bybit public API v5. Each institutional participant is configured with different trading pairs or time windows, simulating the heterogeneous data distribution typical of cross-silo FL.

The Bybit API endpoint used is:

```
GET https://api.bybit.com/v5/market/kline
```

Parameters include symbol (e.g., BTCUSDT), interval (e.g., 60 for 1-hour candles), and limit (number of candles). The fetcher parses the JSON response, extracts OHLCV data, and constructs feature vectors for model training.

Each institutional client derives features from the raw kline data:
- **Returns**: `(close - open) / open`
- **Volatility proxy**: `(high - low) / open`
- **Volume signal**: Normalized volume

The target variable is the next-period return, making this a one-step-ahead prediction task.

In a production scenario, each institution would augment this public data with their proprietary signals -- execution quality metrics, order flow imbalance, internal alpha factors -- creating a setting where federated learning is genuinely advantageous.

## 7. Key Takeaways

1. **Cross-silo FL enables institutional collaboration**: Trading firms can jointly improve predictive models without sharing proprietary data, overcoming the fundamental tension between data sharing and competitive advantage.

2. **Weighted FedAvg is the baseline algorithm**: It is simple, effective, and well-understood. Weighting by dataset size ensures fair representation; modifications like FedProx handle non-IID distributions.

3. **Secure aggregation is essential, not optional**: Without it, the aggregation server can trivially recover individual updates when K is small (typical in cross-silo settings). The pairwise masking protocol provides strong protection against honest-but-curious adversaries.

4. **Differential privacy adds defense in depth**: Even with secure aggregation, the published global model can leak information. DP provides formal guarantees at the cost of some model utility; epsilon between 1-10 is a practical range for trading applications.

5. **Topology matters**: Star topology with secure aggregation is the practical choice for 3-10 institutional participants. Decentralized topologies become relevant as the number of participants grows or when no trusted coordinator exists.

6. **Non-IID data is the norm in finance**: Each institution's data reflects its unique trading strategy, market access, and risk preferences. Algorithms must account for this heterogeneity to converge reliably.

7. **Start with horizontal FL**: Vertical FL (different features, same samples) is more powerful in theory but significantly harder to implement securely. Most practical deployments begin with horizontal FL and evolve as trust and infrastructure mature.

8. **Regulatory alignment**: Cross-silo FL aligns well with financial regulations (GDPR, MiFID II, CCPA) that restrict data sharing. The ability to improve models without data movement is a genuine regulatory advantage.

9. **Rust is well-suited for FL infrastructure**: Its memory safety guarantees, zero-cost abstractions, and strong type system make it ideal for implementing cryptographic protocols and high-performance model training in production trading systems.

10. **The real value is in proprietary signal fusion**: While our example uses public Bybit data, the true power of cross-silo FL in trading emerges when institutions contribute models trained on proprietary, non-overlapping signals -- creating an ensemble effect that no single firm could achieve alone.
