# Chapter 285: GPT Pretraining for Trading

## 1. Introduction

Generative Pre-trained Transformers (GPTs) have revolutionized natural language processing by learning powerful representations through autoregressive pretraining on massive text corpora. The core insight — predicting the next token in a sequence forces a model to learn deep contextual understanding — transfers remarkably well to financial time series. In this chapter, we explore how GPT-style autoregressive pretraining can be applied to trading, treating discretized price movements as a "language" that the model learns to read and predict.

Financial markets generate vast sequences of price data that exhibit complex temporal dependencies, regime changes, and multi-scale patterns. Traditional approaches to modeling these sequences (ARIMA, GARCH, simple RNNs) struggle to capture long-range dependencies and nonlinear interactions. GPT-style architectures, with their causal self-attention mechanism and scalable training paradigm, offer a compelling alternative.

The pretraining-then-fine-tuning paradigm is particularly well-suited to trading applications:

1. **Pretraining phase**: The model learns general market dynamics by predicting the next price token across millions of historical sequences. This builds a foundation of market knowledge — understanding volatility clustering, mean reversion tendencies, momentum effects, and cross-asset correlations.

2. **Fine-tuning phase**: The pretrained model is adapted to specific downstream tasks such as regime classification, directional prediction, or signal generation with a lightweight task-specific head.

This two-phase approach offers several advantages over training from scratch: better sample efficiency on limited labeled data, more robust learned representations, and the ability to transfer knowledge across instruments and timeframes.

## 2. Mathematical Foundations

### 2.1 Causal Language Model Objective

Let $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ be a sequence of tokens. The GPT model factorizes the joint probability using the chain rule:

$$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

The training objective maximizes the log-likelihood:

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

where $\theta$ represents the model parameters. This is equivalent to minimizing the cross-entropy loss between the predicted token distribution and the actual next token.

### 2.2 Causal Self-Attention

The transformer decoder block computes attention as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where $M$ is the causal mask:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

This mask ensures that position $i$ can only attend to positions $j \leq i$, preserving the autoregressive property. Given input embeddings $H \in \mathbb{R}^{T \times d}$, we compute:

$$Q = HW^Q, \quad K = HW^K, \quad V = HW^V$$

with learnable projection matrices $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$.

### 2.3 Multi-Head Attention

Multiple attention heads capture different relationship patterns:

$$\text{MultiHead}(H) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where each $\text{head}_i = \text{Attention}(HW_i^Q, HW_i^K, HW_i^V)$.

### 2.4 Positional Encoding

Since the transformer architecture has no inherent notion of position, we add sinusoidal positional encodings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Alternatively, learnable positional embeddings $E_{pos} \in \mathbb{R}^{T_{max} \times d}$ can be used.

### 2.5 Transformer Decoder Block

Each decoder layer applies:

$$H' = \text{LayerNorm}(H + \text{MultiHead}(H))$$
$$H'' = \text{LayerNorm}(H' + \text{FFN}(H'))$$

where the feed-forward network is:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

## 3. Financial Tokenization

### 3.1 Price Discretization

The first challenge in applying GPT to financial data is converting continuous price series into discrete tokens. We discretize log-returns into a fixed vocabulary:

Given a price series $\{p_t\}$, compute log-returns:

$$r_t = \ln\left(\frac{p_t}{p_{t-1}}\right)$$

Then quantize into $V$ bins using percentile-based boundaries:

$$\text{token}(r_t) = \arg\min_k |r_t - c_k|$$

where $\{c_k\}_{k=1}^{V}$ are bin centers derived from the empirical return distribution. A typical vocabulary might use $V = 256$ bins, providing sufficient granularity to capture meaningful price movements while keeping the vocabulary manageable.

### 3.2 Special Tokens

Beyond return tokens, the vocabulary includes:
- **[BOS]**: Beginning of sequence marker
- **[EOS]**: End of sequence marker
- **[SEP]**: Separator between different instruments or timeframes
- **[MASK]**: Used for optional masked language model auxiliary objectives

### 3.3 Multi-Feature Tokenization

For richer representations, multiple features can be tokenized and interleaved:
- Price returns (primary signal)
- Volume changes (liquidity information)
- Volatility regime indicators
- Time-of-day tokens (for intraday data)

This creates a multi-channel token sequence: $(r_t^{\text{price}}, r_t^{\text{vol}}, r_t^{\text{regime}}, \ldots)$ that captures the full market microstructure.

### 3.4 Vocabulary Construction

The bin boundaries are determined during preprocessing:

1. Collect all returns from the training corpus
2. Compute percentiles $\{q_1, q_2, \ldots, q_{V-1}\}$ evenly spaced from 0% to 100%
3. Bin centers $c_k = (q_k + q_{k-1})/2$
4. Special handling for extreme returns (fat tails): the outermost bins capture all returns beyond the boundary percentiles

This percentile-based approach ensures roughly uniform token frequencies, which improves training stability and prevents the model from being dominated by common small movements.

## 4. Fine-Tuning for Trading

### 4.1 Regime Prediction

After pretraining, a classification head maps the model's hidden states to regime labels:

$$\hat{y}_t = \text{softmax}(W_{\text{cls}} \cdot h_t + b_{\text{cls}})$$

where $h_t$ is the transformer output at position $t$, and regimes might include: trending-up, trending-down, mean-reverting, high-volatility, low-volatility.

### 4.2 Signal Generation

For direct trading signal generation, the fine-tuning head produces continuous signals:

$$s_t = \tanh(W_{\text{sig}} \cdot h_t + b_{\text{sig}})$$

where $s_t \in [-1, 1]$ represents the desired position (short to long). The fine-tuning loss combines prediction accuracy with trading-relevant metrics:

$$\mathcal{L}_{\text{trade}} = -\frac{1}{T}\sum_{t=1}^{T} s_t \cdot r_{t+1} + \lambda \sum_{t=2}^{T} |s_t - s_{t-1}|$$

The second term penalizes excessive position changes (turnover regularization).

### 4.3 Transfer Learning Benefits

Pretraining provides several advantages for the fine-tuning phase:
- **Data efficiency**: Fine-tuning requires far fewer labeled examples than training from scratch
- **Robustness**: Pretrained representations generalize better to unseen market conditions
- **Multi-task capability**: The same pretrained backbone supports multiple downstream tasks
- **Cross-asset transfer**: A model pretrained on one set of instruments can be fine-tuned on others

## 5. Rust Implementation

Our Rust implementation provides an efficient, production-ready GPT pretraining pipeline for trading. The key components are:

### 5.1 Price Tokenizer

The `PriceTokenizer` struct handles the discretization of continuous returns into token indices. It computes bin boundaries from training data using percentile-based quantization and provides both encoding (returns to tokens) and decoding (tokens to approximate returns) functionality.

### 5.2 GPT Decoder

The `GptDecoder` implements a simplified transformer decoder with:
- Causal self-attention with configurable number of heads
- Sinusoidal positional encoding
- Layer normalization
- Feed-forward sublayers with GELU activation
- Next-token prediction via softmax output layer

### 5.3 Training Pipeline

The training loop:
1. Fetches historical OHLCV data from the Bybit API
2. Computes log-returns and tokenizes them
3. Creates overlapping sequences of fixed context length
4. Trains the GPT model using cross-entropy loss with gradient descent
5. Evaluates on held-out data using perplexity and directional accuracy

### 5.4 Generation and Inference

Autoregressive generation produces multi-step forecasts:
1. Feed context window of historical tokens
2. Sample or argmax the next-token distribution
3. Append predicted token to context
4. Repeat for desired forecast horizon

Temperature scaling controls the randomness of generation: lower temperature produces more conservative (peaked) predictions, while higher temperature explores a wider range of scenarios.

## 6. Bybit Data Integration

The implementation connects to the Bybit V5 API to fetch real-time and historical market data:

```
GET https://api.bybit.com/v5/market/kline
Parameters:
  - category: "spot" or "linear"
  - symbol: e.g., "BTCUSDT"
  - interval: candle interval ("1", "5", "15", "60", "D")
  - limit: number of candles (max 200)
```

The API returns OHLCV (Open, High, Low, Close, Volume) data which is processed into:
- Log-returns for tokenization
- Volume changes for auxiliary features
- High-low range for volatility estimation

Data preprocessing includes:
1. **Outlier handling**: Extreme returns beyond configurable thresholds are clipped
2. **Missing data**: Gaps in the time series are forward-filled or flagged with special tokens
3. **Normalization**: Returns are standardized using rolling statistics to handle non-stationarity

## 7. Key Takeaways

1. **GPT-style pretraining transfers to finance**: The autoregressive next-token prediction objective, when applied to discretized financial returns, learns meaningful representations of market dynamics including momentum, mean reversion, and volatility clustering.

2. **Tokenization design is critical**: The choice of vocabulary size, bin boundaries, and multi-feature encoding significantly impacts model quality. Percentile-based quantization ensures balanced token frequencies and stable training.

3. **Causal attention preserves temporal causality**: The masked self-attention mechanism naturally respects the temporal ordering of financial data, preventing future information leakage — a common pitfall in financial ML.

4. **Pretraining enables data-efficient fine-tuning**: Large-scale pretraining on unlabeled price data creates a foundation model that can be quickly adapted to specific trading tasks with limited labeled data.

5. **Autoregressive generation supports scenario analysis**: Beyond point predictions, the generative nature of GPT allows sampling multiple future trajectories, providing a natural framework for risk assessment and uncertainty quantification.

6. **Scalability through Rust**: The Rust implementation provides the performance characteristics needed for real-time inference in production trading systems, with memory safety guarantees that reduce operational risk.

7. **Regularization matters**: Turnover penalties, dropout, and careful learning rate scheduling are essential to prevent overfitting and ensure that fine-tuned models produce tradeable signals rather than noisy predictions.

8. **Cross-asset pretraining enhances generalization**: Models pretrained on diverse instruments develop more robust representations than those trained on a single asset, similar to how language models benefit from diverse text corpora.
