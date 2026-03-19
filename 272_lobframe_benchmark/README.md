# Chapter 272: LOBFrame Benchmark Trading

## 1. Introduction - LOBFrame: A Unified Benchmarking Framework for LOB Prediction

The Limit Order Book (LOB) is the fundamental data structure in modern electronic markets. It records all outstanding buy and sell orders at various price levels, providing a granular view of market supply and demand. Predicting the future state of the LOB — specifically the direction of mid-price movement — has become a central research problem in quantitative finance and machine learning.

However, the field has long suffered from a fragmentation problem. Different research groups use different datasets, different preprocessing pipelines, different evaluation protocols, and different train/test splits. This makes it nearly impossible to perform fair, apples-to-apples comparisons between models. A model that appears to outperform on one benchmark may underperform on another simply due to differences in data normalization or evaluation methodology.

**LOBFrame** addresses this problem by providing a unified benchmarking framework for LOB-based prediction tasks. Inspired by standardized benchmarks in computer vision (ImageNet) and natural language processing (GLUE), LOBFrame establishes:

- **Standardized data pipelines** with consistent normalization and feature engineering
- **Unified evaluation metrics** that capture multiple aspects of prediction quality
- **Reference implementations** of baseline models (LSTM, CNN, Transformer)
- **Reproducible experimental protocols** with fixed train/validation/test splits

The core prediction task in LOBFrame is **mid-price direction classification**: given the current state of the LOB, predict whether the mid-price will go up, stay flat, or go down over a specified prediction horizon. This three-class classification problem is the most widely studied task in LOB literature.

### Why Benchmarking Matters for Trading

In production trading systems, model selection has direct financial consequences. A model that achieves 52% accuracy instead of 51% on a well-calibrated benchmark can translate to significant PnL differences over thousands of trades. Without standardized benchmarks, teams waste resources re-implementing baselines, debugging data pipelines, and running non-comparable experiments.

LOBFrame enables:
- **Fair model comparison**: all models train and evaluate on identical data splits
- **Rapid prototyping**: new models can be tested against established baselines immediately
- **Reproducibility**: results can be verified and extended by other researchers
- **Transfer learning assessment**: models trained on one asset can be evaluated on another using the same framework

## 2. Mathematical Foundation - Standardized Evaluation Metrics

### Mid-Price and Label Construction

The mid-price at time $t$ is defined as:

$$p_t^{mid} = \frac{p_t^{ask} + p_t^{bid}}{2}$$

where $p_t^{ask}$ is the best ask price and $p_t^{bid}$ is the best bid price.

For a prediction horizon $H$, we compute the smoothed future mid-price:

$$\bar{p}_{t+H} = \frac{1}{H} \sum_{i=1}^{H} p_{t+i}^{mid}$$

The label $y_t$ is constructed using a threshold $\theta$:

$$y_t = \begin{cases} +1 & \text{if } \frac{\bar{p}_{t+H} - p_t^{mid}}{p_t^{mid}} > \theta \\ -1 & \text{if } \frac{\bar{p}_{t+H} - p_t^{mid}}{p_t^{mid}} < -\theta \\ 0 & \text{otherwise} \end{cases}$$

### Evaluation Metrics

LOBFrame mandates reporting multiple metrics to capture different aspects of prediction quality:

**Accuracy:**

$$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i = \hat{y}_i]$$

**Precision, Recall, and F1 Score (per class $c$):**

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F_1^c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

The macro-averaged F1 is:

$$F_1^{macro} = \frac{1}{C} \sum_{c=1}^{C} F_1^c$$

**Matthews Correlation Coefficient (MCC):**

MCC is particularly important for LOB prediction because class distributions are often imbalanced. For multi-class problems:

$$\text{MCC} = \frac{\sum_k \sum_l \sum_m C_{kk} C_{lm} - C_{kl} C_{mk}}{\sqrt{\sum_k (\sum_l C_{kl})(\sum_{k' \neq k} \sum_l C_{k'l})} \cdot \sqrt{\sum_k (\sum_l C_{lk})(\sum_{k' \neq k} \sum_l C_{lk'})}}$$

where $C$ is the confusion matrix. MCC ranges from $-1$ to $+1$, where $+1$ indicates perfect prediction, $0$ indicates random prediction, and $-1$ indicates total disagreement.

### Z-Score Normalization

LOBFrame standardizes the normalization procedure using z-score normalization per feature:

$$\hat{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$, computed on the training set only (to prevent data leakage).

## 3. Benchmark Models

### 3.1 Baseline LSTM

The LSTM baseline processes the LOB snapshot as a sequence of feature vectors. For an LOB with $L$ levels on each side, the input at each timestep is a vector of dimension $4L$ (bid prices, bid volumes, ask prices, ask volumes).

**Architecture:**
- Input layer: $4L$-dimensional feature vector
- LSTM layer: hidden size 64, single layer
- Fully connected layer: 64 -> 3 (three classes)
- Softmax output

The LSTM captures temporal dependencies in the LOB state sequence, learning patterns such as order flow imbalance building up before a price move.

### 3.2 Baseline CNN

The CNN baseline treats the LOB snapshot as a 2D image-like structure, where one axis represents the price level and the other represents the feature type (price vs. volume, bid vs. ask).

**Architecture:**
- Input: $(T, 4L)$ matrix reshaped to $(T, 4, L)$
- Conv2D layer: 16 filters, kernel $(3, 3)$, ReLU
- Conv2D layer: 32 filters, kernel $(3, 3)$, ReLU
- Global average pooling
- Fully connected: 32 -> 3
- Softmax output

CNNs excel at capturing local spatial patterns in the LOB, such as large orders at specific price levels or symmetric/asymmetric book shapes.

### 3.3 Baseline Transformer

The Transformer baseline uses self-attention to model interactions between different parts of the LOB without the sequential bottleneck of LSTM.

**Architecture:**
- Input embedding: linear projection of $4L$ features to $d_{model} = 64$
- Positional encoding (sinusoidal)
- 2 Transformer encoder layers, 4 attention heads
- Classification head: mean pooling -> linear -> 3 classes

The Transformer can directly attend to relevant price levels regardless of their position in the sequence, potentially capturing long-range dependencies more efficiently.

## 4. Data Normalization and Feature Engineering Standards

### 4.1 Standard Feature Set

LOBFrame defines a canonical feature set for LOB representation:

| Feature Group | Features | Description |
|---|---|---|
| Raw Prices | $p_1^{bid}, ..., p_L^{bid}, p_1^{ask}, ..., p_L^{ask}$ | Bid/ask prices at L levels |
| Raw Volumes | $v_1^{bid}, ..., v_L^{bid}, v_1^{ask}, ..., v_L^{ask}$ | Bid/ask volumes at L levels |
| Spread | $s = p_1^{ask} - p_1^{bid}$ | Best bid-ask spread |
| Mid-Price | $p^{mid} = (p_1^{ask} + p_1^{bid}) / 2$ | Mid-price |
| Order Imbalance | $OI_l = \frac{v_l^{bid} - v_l^{ask}}{v_l^{bid} + v_l^{ask}}$ | Volume imbalance per level |
| Price Differences | $\Delta p_l = p_{l+1} - p_l$ | Price gaps between levels |

### 4.2 Normalization Protocol

1. **Training set statistics**: compute mean and standard deviation per feature on the training set
2. **Apply z-score**: normalize all sets (train, validation, test) using training set statistics
3. **Clipping**: clip normalized values to $[-5, +5]$ to handle extreme outliers
4. **No future information**: normalization statistics are never computed using validation or test data

### 4.3 Temporal Windowing

LOBFrame uses a sliding window approach:
- Window size $T$: number of consecutive LOB snapshots (default: 100)
- Step size $S$: stride between windows (default: 1 for training, $T$ for testing)
- Prediction horizon $H$: number of steps ahead for label construction (default: 10)

## 5. Rust Implementation

Our Rust implementation provides a high-performance LOBFrame benchmark runner. The key components are:

- **`LobNormalizer`**: z-score normalization with training-set statistics, including clipping
- **`EvaluationMetrics`**: accuracy, macro F1, and MCC computation from confusion matrices
- **`BaselineLstm` / `BaselineCnn`**: simplified forward-pass implementations for benchmarking
- **`BenchmarkRunner`**: orchestrates model evaluation with standardized protocol
- **`BybitOrderbook`**: fetches live LOB data from Bybit exchange

The implementation uses `ndarray` for numerical operations and `reqwest` for HTTP calls to the Bybit API. All models implement a common `BenchmarkModel` trait, ensuring uniform interface across different architectures.

### Design Decisions

- **Trait-based architecture**: all models implement `BenchmarkModel` with `predict()` and `name()` methods
- **Deterministic evaluation**: fixed random seeds for reproducible results
- **Streaming normalization**: normalizer can be updated incrementally for online settings
- **Zero-copy where possible**: LOB data structures avoid unnecessary allocations

See `rust/src/lib.rs` for the full implementation and `rust/examples/trading_example.rs` for a complete usage example.

## 6. Bybit Data Integration

LOBFrame integrates with the Bybit cryptocurrency exchange to provide real-time LOB data for benchmarking. The Bybit API provides:

- **Orderbook snapshots**: up to 200 levels of bid/ask prices and quantities
- **Low latency**: REST API for snapshot data, WebSocket for streaming
- **Multiple assets**: BTCUSDT, ETHUSDT, and other perpetual futures contracts

### Integration Architecture

```
Bybit REST API (/v5/market/orderbook)
        |
        v
  Raw JSON Response
        |
        v
  BybitOrderbook Parser (Rust serde)
        |
        v
  LOB Feature Extraction
        |
        v
  Z-Score Normalization (LobNormalizer)
        |
        v
  Model Input (ndarray Array2)
        |
        v
  Benchmark Runner (predict + evaluate)
```

The Bybit integration allows traders to:
1. Fetch live orderbook data for any supported trading pair
2. Apply the same normalization pipeline used in research benchmarks
3. Run trained models on live data with consistent preprocessing
4. Compare model performance on real market data vs. historical benchmarks

### API Usage

The implementation uses Bybit's v5 market data endpoint:
- Endpoint: `GET /v5/market/orderbook`
- Parameters: `category=linear`, `symbol=BTCUSDT`, `limit=50`
- No authentication required for public market data

## 7. Key Takeaways

1. **Standardized benchmarks are essential**: without common evaluation protocols, comparing LOB prediction models is meaningless. LOBFrame provides the infrastructure for fair comparison.

2. **Multiple metrics matter**: accuracy alone is insufficient for imbalanced LOB data. F1 and MCC provide complementary views of model quality, especially when class distributions are skewed.

3. **Normalization is critical**: z-score normalization with training-set-only statistics prevents data leakage and ensures models generalize properly. Small differences in normalization can lead to large differences in reported accuracy.

4. **Baselines set the bar**: LSTM, CNN, and Transformer baselines establish minimum performance thresholds. Any new model should demonstrably outperform these baselines on the standardized benchmark.

5. **Live data integration bridges research and production**: by using the same framework for both historical benchmarks and live Bybit data, LOBFrame ensures that research results translate to production performance.

6. **Rust enables production-grade performance**: the Rust implementation provides the speed necessary for real-time LOB processing while maintaining the correctness guarantees needed for financial systems.

7. **Reproducibility drives progress**: fixed splits, deterministic evaluation, and open-source implementations ensure that results can be verified and built upon by the community.

## References

- Lucchese, L., Pakkanen, M. S., & Veraart, A. E. D. (2024). "LOBFrame: A framework for studying limit order book models." *Quantitative Finance*.
- Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." *IEEE Transactions on Signal Processing*.
- Ntakaris, A., Magris, M., Stoll, J., Kanber, J., & Iosifidis, A. (2018). "Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods." *Journal of Forecasting*.
- Tran, D. T., Iosifidis, A., Kanniainen, J., & Gabbouj, M. (2018). "Temporal attention-augmented bilinear network for financial time-series data analysis." *IEEE Transactions on Neural Networks and Learning Systems*.
