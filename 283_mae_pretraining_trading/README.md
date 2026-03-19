# Chapter 283: MAE (Masked Autoencoder) Pretraining for Trading

## 1. Introduction

Masked Autoencoders (MAE), introduced by He et al. (2022), represent a paradigm shift in self-supervised learning. Originally designed for computer vision, the core idea is strikingly simple yet powerful: mask a large portion (75% or more) of input patches and train a model to reconstruct the missing content. This forces the encoder to learn rich, semantic representations of the underlying data rather than memorizing superficial patterns.

The MAE approach draws inspiration from the success of masked language modeling in NLP (e.g., BERT), but adapts it to continuous signals with an asymmetric encoder-decoder architecture. The encoder operates only on visible (unmasked) patches, making pretraining computationally efficient, while a lightweight decoder reconstructs the full input from the latent representation plus mask tokens.

For financial markets, MAE pretraining offers a compelling framework. Market data --- OHLCV (Open, High, Low, Close, Volume) time series --- contains rich temporal structure, cross-asset correlations, and regime-dependent dynamics. By masking substantial portions of market data and training a model to reconstruct them, we can learn representations that capture market microstructure, volatility clustering, mean reversion, momentum, and other phenomena without requiring labeled data.

This is particularly valuable in trading because labeled data (e.g., regime labels, profitable trade signals) is scarce and expensive to produce, while raw market data is abundant. MAE pretraining enables us to leverage vast quantities of unlabeled historical data to build powerful feature extractors that can then be fine-tuned for specific downstream tasks such as regime detection, anomaly detection, return prediction, and risk assessment.

### Key advantages of MAE for trading:
- **Data efficiency**: Learns from unlabeled market data, which is abundant
- **Robust representations**: High masking ratio prevents shortcut learning
- **Computational efficiency**: Encoder processes only visible patches (25% of input)
- **Transfer learning**: Pretrained encoder transfers to multiple downstream tasks
- **Market microstructure learning**: Reconstructing masked windows forces understanding of price dynamics

## 2. Mathematical Foundation

### 2.1 Patch Embedding for Time Series

Given a multivariate time series $X \in \mathbb{R}^{T \times D}$ where $T$ is the number of time steps and $D$ is the feature dimension (e.g., 5 for OHLCV), we divide the series into non-overlapping patches:

$$X = [x_1, x_2, \ldots, x_N], \quad x_i \in \mathbb{R}^{P \times D}$$

where $P$ is the patch size (number of time steps per patch) and $N = \lfloor T/P \rfloor$ is the number of patches. Each patch is linearly projected to a $d$-dimensional embedding:

$$e_i = W_e \cdot \text{flatten}(x_i) + b_e, \quad e_i \in \mathbb{R}^d$$

where $W_e \in \mathbb{R}^{d \times (P \cdot D)}$ is the projection matrix.

### 2.2 Random Masking Strategy

We randomly select a subset of patches to mask. Let $\mathcal{M} \subset \{1, 2, \ldots, N\}$ denote the set of masked patch indices, with $|\mathcal{M}| = \lfloor r \cdot N \rfloor$ where $r$ is the masking ratio (typically 0.75).

The visible set is $\mathcal{V} = \{1, \ldots, N\} \setminus \mathcal{M}$, containing only $(1-r) \cdot N$ patches.

A high masking ratio is critical: it prevents the model from simply interpolating between nearby visible patches and forces it to learn genuine semantic understanding of the data generating process.

### 2.3 Asymmetric Encoder-Decoder

**Encoder**: A transformer that processes only the visible patches:

$$H = \text{Encoder}(\{e_i + p_i : i \in \mathcal{V}\})$$

where $p_i$ is a positional embedding for patch $i$. The encoder is the heavy component (deep, wide transformer) but operates on only 25% of patches, yielding significant computational savings.

**Decoder**: A lightweight transformer that takes the full set of tokens --- encoded visible patches plus learnable mask tokens $m \in \mathbb{R}^d$ at masked positions:

$$Z = \text{Decoder}([h_1, h_2, \ldots, h_N])$$

where $h_i = H_i$ if $i \in \mathcal{V}$, and $h_i = m + p_i$ if $i \in \mathcal{M}$.

### 2.4 Reconstruction Loss

The loss is computed only on masked patches using Mean Squared Error:

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|x_i - \hat{x}_i\|_2^2$$

where $\hat{x}_i$ is the decoder's reconstruction of patch $i$. Normalizing each patch (subtracting mean, dividing by standard deviation) before computing the loss improves training stability, especially for financial data with varying scales.

### 2.5 Why High Masking Ratio Works

In financial time series, adjacent patches are highly correlated (prices are continuous, volatility clusters). With low masking ratios (e.g., 25%), the model can trivially interpolate. At 75% masking, the model must:

1. Understand the underlying volatility regime
2. Infer trend direction from sparse observations
3. Model the joint distribution of OHLCV features
4. Capture long-range dependencies across the visible patches

This creates a challenging pretext task that yields rich representations.

## 3. Financial MAE: Masking OHLCV Time Windows

### 3.1 Data Representation

For financial MAE, we structure OHLCV data into a 2D representation where:
- **Rows**: Time steps (e.g., 1-minute, 5-minute, or daily bars)
- **Columns**: Features (Open, High, Low, Close, Volume)

Each patch captures a window of $P$ consecutive bars across all features, providing a local view of price action.

### 3.2 Preprocessing

Financial data requires careful normalization:

1. **Log returns**: Convert prices to log returns for stationarity: $r_t = \ln(P_t / P_{t-1})$
2. **Volume normalization**: Apply log transform and z-score normalization
3. **Per-patch normalization**: Subtract patch mean and divide by patch standard deviation before computing reconstruction loss
4. **Handling missing data**: Mark truly missing data separately from artificial masking

### 3.3 Learning Market Microstructure

By reconstructing masked OHLCV windows, the MAE learns:

- **Volatility patterns**: Reconstructing price ranges requires understanding volatility clustering (GARCH-like dynamics)
- **Volume-price relationships**: The model learns that high volume often accompanies large price moves
- **Intraday patterns**: For high-frequency data, the model captures U-shaped volume curves and opening/closing dynamics
- **Cross-feature dependencies**: OHLC relationships encode candlestick patterns, which the model must understand for accurate reconstruction
- **Trend continuation/reversal**: Reconstructing future patches from past visible ones requires understanding momentum and mean reversion

### 3.4 Masking Strategies for Finance

Beyond uniform random masking, financial-specific strategies include:

- **Block masking**: Mask contiguous time blocks to force longer-range prediction
- **Feature masking**: Mask specific features (e.g., all volume data) to learn cross-feature dependencies
- **Regime-aware masking**: Increase masking during volatile periods where reconstruction is harder
- **Multi-scale masking**: Apply different masking ratios at different temporal scales

## 4. Transfer to Downstream Tasks

### 4.1 Regime Detection

The pretrained encoder captures latent market states. Fine-tuning for regime detection:

1. Freeze the MAE encoder (or use low learning rate)
2. Add a classification head: $\hat{y} = \text{softmax}(W_c \cdot h_{\text{[CLS]}} + b_c)$
3. Train on labeled regime data (bull/bear/sideways)

The pretrained representations typically require 10x fewer labeled samples than training from scratch.

### 4.2 Anomaly Detection

The reconstruction error itself serves as an anomaly score:

$$\text{anomaly\_score}(X) = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2$$

Anomalous market conditions (flash crashes, circuit breakers, extreme events) produce high reconstruction errors because they deviate from learned normal patterns. This provides a natural, unsupervised anomaly detector.

### 4.3 Return Prediction

Fine-tune the encoder for next-window return prediction:

1. Use the encoder's output representation
2. Add a regression head for predicting returns over the next $k$ periods
3. The pretrained features capture market dynamics that improve prediction accuracy

### 4.4 Risk Assessment

The decoder's uncertainty in reconstruction correlates with market uncertainty:
- Patches that are consistently hard to reconstruct indicate unpredictable market regimes
- This provides a data-driven volatility/risk indicator

## 5. Rust Implementation

Our Rust implementation provides a complete MAE framework for trading applications. The key components are:

### Architecture Overview

```
Input OHLCV Series
       |
  [Patch Embedding] -- Split into P-sized windows, project to d dimensions
       |
  [Random Masking] -- Remove 75% of patches
       |
  [Encoder] -- Process visible patches (transformer layers)
       |
  [Insert Mask Tokens]
       |
  [Decoder] -- Reconstruct all patches (lightweight transformer)
       |
  [MSE Loss] -- Only on masked patches
```

The implementation in `rust/src/lib.rs` includes:
- `PatchEmbedding`: Converts OHLCV windows into dense embeddings
- `RandomMasker`: Implements the 75% masking strategy
- `MAEEncoder`: Transformer encoder for visible patches
- `MAEDecoder`: Lightweight decoder for reconstruction
- `MaskedAutoencoder`: Full MAE pipeline with training loop
- `BybitClient`: Fetches real-time and historical market data

### Performance Considerations

Rust provides several advantages for MAE training on financial data:
- Zero-cost abstractions for matrix operations via `ndarray`
- Memory safety without garbage collection pauses
- Easy parallelization across patches
- Low-latency data ingestion from exchange APIs

## 6. Bybit Data Integration

The implementation integrates with the Bybit exchange API to fetch real market data:

### API Endpoints Used
- `GET /v5/market/kline` - Historical OHLCV candlestick data
- Supports multiple intervals: 1m, 5m, 15m, 1h, 4h, 1d
- Returns up to 1000 candles per request

### Data Pipeline

```
Bybit API --> Raw OHLCV --> Log Returns --> Patch Creation --> MAE Training
```

The pipeline handles:
1. **Rate limiting**: Respects API rate limits with configurable delays
2. **Data validation**: Checks for gaps, zeros, and anomalies in raw data
3. **Normalization**: Applies log returns and z-score normalization
4. **Windowing**: Creates overlapping or non-overlapping patches for training

### Example Usage

```rust
let client = BybitClient::new();
let klines = client.fetch_klines("BTCUSDT", "15", 1000).await?;
let patches = create_patches(&klines, patch_size);
let mae = MaskedAutoencoder::new(config);
mae.pretrain(&patches, epochs);
```

## 7. Key Takeaways

1. **MAE pretraining is a powerful self-supervised method** for learning representations from unlabeled financial data. By masking 75%+ of input patches and training to reconstruct them, the model learns deep understanding of market dynamics.

2. **The asymmetric encoder-decoder architecture is computationally efficient**: the heavy encoder processes only 25% of patches, while a lightweight decoder handles reconstruction. This makes pretraining on large historical datasets feasible.

3. **High masking ratios are essential for financial data** because adjacent time windows are highly correlated. Without aggressive masking, the model learns trivial interpolation rather than meaningful representations.

4. **Financial MAE captures market microstructure**: volatility clustering, volume-price relationships, trend dynamics, and regime-dependent behavior emerge naturally from the reconstruction objective.

5. **Transfer learning to downstream tasks is highly effective**: the pretrained encoder can be fine-tuned for regime detection, anomaly detection, return prediction, and risk assessment with significantly less labeled data.

6. **Reconstruction error serves as a natural anomaly detector**: unusual market conditions produce high reconstruction loss, providing an unsupervised monitoring signal.

7. **Rust implementation enables production-grade performance**: zero-cost abstractions, memory safety, and low-latency API integration make Rust ideal for deploying MAE-based trading systems.

8. **Integration with exchange APIs (Bybit)** enables continuous model updating with fresh market data, keeping representations current as market dynamics evolve.

## References

- He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.
- Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Zerveas, G., et al. (2021). A Transformer-based Framework for Multivariate Time Series Representation Learning. KDD 2021.
- Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023.
