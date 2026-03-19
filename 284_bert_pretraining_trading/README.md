# Chapter 284: BERT Pretraining for Trading

## Introduction

Bidirectional Encoder Representations from Transformers (BERT) introduced a paradigm shift in natural language processing by demonstrating that large-scale unsupervised pretraining on text corpora, followed by task-specific fine-tuning, produces state-of-the-art results across a wide range of downstream tasks. The core insight behind BERT is simple yet powerful: by training a model to predict randomly masked tokens from their bidirectional context, the model learns deep contextual representations that capture syntactic, semantic, and pragmatic information about language.

In the context of financial markets, this same principle can be applied in two complementary directions. First, financial text data --- earnings reports, analyst notes, news articles, SEC filings, and social media posts --- contains rich information that directly impacts asset prices. Models pretrained on financial text corpora (such as FinBERT) have been shown to significantly outperform general-purpose language models on sentiment analysis, event detection, and market impact prediction. Second, and perhaps more innovatively, the masked pretraining paradigm can be adapted to numerical financial time series by treating discretized price movements, volume bins, and technical indicator levels as "tokens" in a sequence, then training the model to predict masked values from their temporal context.

This chapter provides a comprehensive treatment of BERT-style pretraining for trading applications. We begin with the mathematical foundations of masked language modeling and bidirectional attention, then explore financial adaptations including FinBERT, price tokenization strategies, and regime-aware next sentence prediction. We present fine-tuning approaches for sentiment analysis, event classification, and trend prediction. Finally, we provide a complete Rust implementation with Bybit exchange integration for real-time cryptocurrency trading.

## Mathematical Foundations

### Masked Language Model (MLM)

The MLM objective is the cornerstone of BERT pretraining. Given a sequence of tokens $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, we randomly select a subset $\mathcal{M} \subset \{1, 2, \ldots, n\}$ of positions to mask (typically 15% of all tokens). The masked tokens are replaced according to a stochastic policy:

- With probability 0.8, replace $x_i$ with the special `[MASK]` token.
- With probability 0.1, replace $x_i$ with a random token from the vocabulary.
- With probability 0.1, keep $x_i$ unchanged.

The MLM loss is then defined as:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \mathbf{x}_{\setminus \mathcal{M}}; \theta)$$

where $\mathbf{x}_{\setminus \mathcal{M}}$ denotes the sequence with masked positions and $\theta$ represents the model parameters. The conditional probability is computed via a softmax over the vocabulary:

$$P(x_i \mid \mathbf{x}_{\setminus \mathcal{M}}; \theta) = \frac{\exp(\mathbf{e}_{x_i}^\top \mathbf{h}_i)}{\sum_{v \in \mathcal{V}} \exp(\mathbf{e}_v^\top \mathbf{h}_i)}$$

where $\mathbf{h}_i$ is the hidden representation at position $i$ produced by the encoder, and $\mathbf{e}_v$ is the embedding vector for vocabulary token $v$.

### Next Sentence Prediction (NSP)

The NSP objective trains the model to understand relationships between pairs of sequences. Given two segments $A$ and $B$, the model predicts whether $B$ is the actual next segment following $A$ in the corpus, or a randomly sampled segment:

$$\mathcal{L}_{\text{NSP}} = -[y \log P(\text{IsNext} \mid A, B; \theta) + (1 - y) \log P(\text{NotNext} \mid A, B; \theta)]$$

where $y = 1$ if $B$ is the true continuation of $A$ and $y = 0$ otherwise. The prediction is made using the `[CLS]` token representation passed through a binary classification head.

### Bidirectional Multi-Head Attention

Unlike autoregressive models (GPT) that attend only to left context, BERT uses bidirectional self-attention. For each attention head $h$, we compute:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q = XW_Q^h$, $K = XW_K^h$, $V = XW_V^h$ are linear projections of the input $X$. Multi-head attention concatenates $H$ heads:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

Each transformer layer applies multi-head attention followed by a position-wise feed-forward network with layer normalization and residual connections:

$$\mathbf{h}' = \text{LayerNorm}(\mathbf{h} + \text{MultiHead}(\mathbf{h}))$$
$$\mathbf{h}'' = \text{LayerNorm}(\mathbf{h}' + \text{FFN}(\mathbf{h}'))$$

The total pretraining loss combines both objectives:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

## Financial BERT: Adapting BERT for Markets

### FinBERT and Domain-Specific Pretraining

FinBERT extends the BERT architecture by continuing pretraining on financial text corpora. The key insight is that financial language has domain-specific vocabulary (e.g., "bearish," "breakout," "consolidation"), unique syntactic patterns, and contextual meanings that differ from general English. For example, the word "volume" in general text refers to loudness or quantity, while in finance it specifically refers to trading volume.

Domain adaptation involves two stages:

1. **Vocabulary extension**: Add financial tokens to the wordpiece vocabulary (e.g., ticker symbols, financial acronyms like EPS, P/E, EBITDA).
2. **Continued pretraining**: Train on financial corpora including 10-K filings, earnings call transcripts, financial news from Reuters and Bloomberg, and analyst reports.

### Masking Price Tokens

For numerical financial time series, we first discretize continuous values into token bins. Given a price series $(p_1, p_2, \ldots, p_T)$, we compute returns $r_t = (p_t - p_{t-1}) / p_{t-1}$ and quantize them into $B$ bins:

$$\text{token}(r_t) = \text{bin}_k \quad \text{where} \quad q_{k-1} \leq r_t < q_k$$

where $q_0 < q_1 < \cdots < q_B$ are quantile boundaries estimated from historical data. Volume, volatility, and other features are similarly tokenized. The masking strategy for price tokens follows the standard MLM protocol, but with domain-specific modifications:

- **Contiguous masking**: Mask consecutive blocks of 2-5 tokens to force the model to learn temporal dynamics rather than simple interpolation.
- **Feature-aligned masking**: When masking a price token at time $t$, optionally mask the corresponding volume and volatility tokens to prevent information leakage.
- **Regime-aware masking**: Increase masking probability during high-volatility periods to force the model to learn representations that are robust to regime changes.

### NSP for Regime Transitions

The standard NSP task is adapted for financial markets by defining "sentences" as market regimes or time windows. Two consecutive windows from the same regime are labeled as `IsNext`, while windows from different regimes (e.g., a bull market segment paired with a crash segment) are labeled as `NotNext`. This forces the model to learn representations that capture regime transitions, which are among the most valuable signals for trading.

## Fine-Tuning Tasks

### Sentiment Analysis

After pretraining, a classification head is added on top of the `[CLS]` representation to predict sentiment (positive, negative, neutral) from financial text. The fine-tuning loss is:

$$\mathcal{L}_{\text{sentiment}} = -\sum_{c \in \{pos, neg, neu\}} y_c \log \hat{y}_c$$

Financial sentiment analysis presents unique challenges: a statement like "revenue declined less than expected" is actually positive in financial context despite containing the negative word "declined."

### Event Classification

The model is fine-tuned to classify market-moving events from text: earnings surprises, mergers and acquisitions, regulatory actions, macroeconomic announcements, and geopolitical events. Multi-label classification enables detection of multiple event types within a single document.

### Trend Prediction

For price-tokenized time series, fine-tuning for trend prediction replaces the MLM head with a classification head that predicts future price direction (up, down, sideways) from the `[CLS]` representation. The model's bidirectional context allows it to consider both recent momentum and historical support/resistance levels when making predictions.

## Rust Implementation

The Rust implementation in this chapter provides a complete BERT pretraining and fine-tuning pipeline. The key components include:

- **PriceTokenizer**: Converts continuous price data into discrete tokens using quantile-based binning, with configurable masking strategies for MLM pretraining.
- **BertEncoder**: Implements multi-head bidirectional self-attention with layer normalization and residual connections.
- **MLMHead**: Maps encoder outputs back to token probabilities for masked token prediction.
- **ClassificationHead**: Provides a fine-tuning layer that maps `[CLS]` representations to trading signals (buy, sell, hold).
- **Bybit API integration**: Fetches real-time OHLCV data from the Bybit exchange for cryptocurrency markets.

The implementation emphasizes correctness and pedagogical clarity. While production deployment would require GPU acceleration (via `tch-rs` or `candle`), our CPU-based ndarray implementation makes the mathematical operations transparent and easy to follow.

```rust
// Example: Tokenize and mask price data
let tokenizer = PriceTokenizer::new(num_bins, mask_ratio);
let tokens = tokenizer.tokenize(&prices);
let (masked_tokens, mask_positions) = tokenizer.mask_tokens(&tokens);

// Encode with BERT
let encoder = BertEncoder::new(config);
let hidden_states = encoder.forward(&masked_tokens);

// Predict masked tokens
let mlm_head = MLMHead::new(hidden_dim, vocab_size);
let predictions = mlm_head.forward(&hidden_states, &mask_positions);
```

## Bybit Data Integration

The implementation connects to Bybit's public REST API to fetch historical kline (candlestick) data. The `/v5/market/kline` endpoint provides OHLCV data at configurable intervals (1m, 5m, 15m, 1h, 4h, 1d). The fetched data is tokenized into discrete bins representing price movements and volume levels, then used for BERT pretraining.

Key considerations for live data integration:

- **Rate limiting**: Bybit allows up to 120 requests per minute; the implementation includes built-in throttling.
- **Data normalization**: Prices are converted to log-returns before tokenization to ensure stationarity.
- **Missing data handling**: Gaps in kline data (e.g., during exchange maintenance) are filled with `[PAD]` tokens.

## Key Takeaways

1. **BERT's masked pretraining paradigm transfers powerfully to financial domains**, both for text-based tasks (sentiment, event classification) and numerical time series analysis (price prediction, regime detection).

2. **Domain-specific pretraining is essential**: Models pretrained on financial text significantly outperform general-purpose BERT on all financial NLP tasks. FinBERT demonstrates that continued pretraining on domain-specific corpora yields substantial improvements.

3. **Price tokenization enables sequence modeling of numerical data**: By discretizing continuous price movements into tokens, we can apply the full power of transformer architectures to time series forecasting.

4. **Bidirectional attention captures market context that autoregressive models miss**: The ability to condition on both past and future context (during pretraining) leads to richer representations that capture support/resistance levels, mean-reversion patterns, and regime structures.

5. **Fine-tuning efficiency**: Once pretrained, BERT can be adapted to diverse downstream tasks with minimal labeled data, making it particularly valuable in finance where labeled datasets are scarce and expensive to create.

6. **Regime-aware NSP provides unique value**: Adapting the next sentence prediction task to detect regime transitions gives the model an explicit mechanism for learning about structural breaks in market behavior.

7. **Production considerations**: While our Rust implementation focuses on correctness and clarity, production deployment requires GPU acceleration, distributed training for large corpora, and careful attention to data freshness and latency for real-time trading applications.

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). FinBERT: A Pretrained Language Model for Financial Communications. arXiv:2006.08097.
- Liu, Z., et al. (2021). FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining. IJCAI.
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
