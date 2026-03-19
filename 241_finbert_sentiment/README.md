# Chapter 241: FinBERT Sentiment

## Introduction

FinBERT is a pre-trained language model specifically fine-tuned for financial sentiment analysis. Built on top of BERT (Bidirectional Encoder Representations from Transformers), FinBERT understands the nuanced language of financial texts — earnings reports, analyst notes, news headlines, and social media posts — and classifies them as positive, negative, or neutral. Unlike general-purpose sentiment tools that struggle with domain-specific vocabulary (where "volatile" can be neutral and "restructuring" can be negative), FinBERT has been trained on thousands of financial documents to capture the subtle meaning behind market-relevant language.

Sentiment analysis has become a critical component of modern quantitative trading. Research has shown that news sentiment can predict short-term price movements, earnings surprises, and volatility spikes. By systematically quantifying the tone of financial text, traders can build alpha signals that complement traditional technical and fundamental analysis. FinBERT brings state-of-the-art NLP accuracy to this task, achieving over 90% accuracy on standard financial sentiment benchmarks.

This chapter presents a complete FinBERT sentiment analysis framework. We cover the architecture of the model, the mechanics of sentiment scoring, strategies for incorporating sentiment into trading signals, and a working Rust implementation that connects to the Bybit cryptocurrency exchange for real-time sentiment-driven trading.

## Key Concepts

### BERT Architecture Foundation

FinBERT inherits its architecture from BERT, which uses a multi-layer bidirectional Transformer encoder. The key innovation of BERT is its pre-training strategy: it learns language representations by jointly conditioning on both left and right context in all layers.

The input to BERT is a sequence of tokens $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, which are converted to embeddings and passed through $L$ Transformer layers. Each layer applies multi-head self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, $V$ are query, key, and value matrices derived from the input, and $d_k$ is the dimension of the key vectors. The multi-head mechanism runs $h$ attention operations in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

The output of the final Transformer layer for the [CLS] token serves as a sentence-level representation that captures the overall meaning of the input text.

### Financial Domain Adaptation

FinBERT adapts the general BERT model to the financial domain through a two-stage process:

1. **Domain pre-training**: The base BERT model is further pre-trained on a large corpus of financial text (e.g., Reuters TRC2 dataset with ~46,000 financial articles). This stage teaches the model financial vocabulary and language patterns.
2. **Task fine-tuning**: The domain-adapted model is fine-tuned on a labeled financial sentiment dataset (e.g., Financial PhraseBank with ~4,845 sentences labeled by financial experts).

The final classification layer maps the [CLS] representation to sentiment probabilities:

$$P(y | \mathbf{x}) = \text{softmax}(W_c \cdot \mathbf{h}_{[CLS]} + b_c)$$

where $y \in \{\text{positive}, \text{negative}, \text{neutral}\}$, $W_c$ is the classification weight matrix, and $\mathbf{h}_{[CLS]}$ is the hidden state of the [CLS] token from the final layer.

### Sentiment Score

For trading applications, the three-class probability distribution is typically converted to a continuous sentiment score:

$$S = P(\text{positive}) - P(\text{negative})$$

This score ranges from -1 (strongly negative) to +1 (strongly positive). A score near 0 indicates neutral sentiment or ambiguity. The continuous score is more useful than discrete labels because it captures the intensity of sentiment, which correlates with the magnitude of subsequent price movements.

### Aggregated Sentiment Signals

Individual text sentiment scores are aggregated across multiple sources and time windows to construct robust trading signals:

$$\bar{S}_t = \frac{1}{N_t} \sum_{i=1}^{N_t} S_i$$

where $N_t$ is the number of texts observed in the time window ending at $t$. Weighted variants can assign higher importance to more recent texts or higher-confidence predictions:

$$\bar{S}_t^w = \frac{\sum_{i=1}^{N_t} w_i \cdot S_i}{\sum_{i=1}^{N_t} w_i}$$

where weights $w_i$ can be based on recency, source reliability, or prediction confidence (the maximum class probability).

## ML Approaches

### Sentiment-Based Alpha Signals

The simplest trading application uses the sentiment score directly as an alpha signal. Given aggregated sentiment $\bar{S}_t$ and a threshold $\theta$:

- **Long signal**: $\bar{S}_t > \theta$ (bullish sentiment)
- **Short signal**: $\bar{S}_t < -\theta$ (bearish sentiment)
- **No trade**: $|\bar{S}_t| \leq \theta$ (neutral or uncertain)

The threshold $\theta$ controls the aggressiveness of the strategy. Higher thresholds produce fewer but higher-conviction trades.

### Sentiment Momentum

Sentiment momentum captures the rate of change in sentiment, which can be more predictive than the level:

$$\Delta S_t = \bar{S}_t - \bar{S}_{t-k}$$

A positive sentiment momentum (improving sentiment) often precedes price appreciation, even if the absolute sentiment level is still negative. This is analogous to the "second derivative" approach used in economic indicators.

### Multi-Factor Integration

Sentiment is most powerful when combined with other factors. A multi-factor model might weight:

$$\alpha_t = \beta_1 \cdot \bar{S}_t + \beta_2 \cdot \Delta S_t + \beta_3 \cdot \text{Volume}_t + \beta_4 \cdot \text{Momentum}_t$$

The coefficients $\beta_i$ are estimated via regression or machine learning on historical data. Sentiment often adds orthogonal information to price-based factors, improving the Sharpe ratio of the combined strategy.

### Logistic Regression for Sentiment Classification

For building a lightweight sentiment classifier without a full Transformer model, logistic regression on TF-IDF or word embedding features provides a practical baseline:

$$P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Training minimizes binary cross-entropy:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

## Feature Engineering

### Text Preprocessing for Financial Documents

Financial text requires specialized preprocessing:

- **Ticker normalization**: Replace ticker symbols ($AAPL, BTC, ETH) with standardized tokens
- **Number handling**: Normalize financial figures (revenue, EPS, market cap) into categorical ranges
- **Negation scope**: Track negation words ("not profitable", "failed to meet") that flip sentiment
- **Domain stopwords**: Remove financial boilerplate ("forward-looking statements", "past performance")

### Keyword-Based Sentiment Features

Before deep learning, keyword-based approaches used financial sentiment dictionaries such as the Loughran-McDonald dictionary, which classifies words into categories:

- **Positive**: profit, growth, improvement, outperform, upgrade
- **Negative**: loss, decline, risk, downgrade, deficit, lawsuit
- **Uncertainty**: may, possible, uncertain, approximately, risk
- **Litigious**: lawsuit, court, legal, regulation, compliance

The ratio of positive to negative words provides a simple but effective sentiment proxy:

$$S_{LM} = \frac{N_{positive} - N_{negative}}{N_{total}}$$

### Sentiment Decay

Sentiment impact on prices decays over time. An exponentially weighted sentiment score captures this:

$$\hat{S}_t = \sum_{i=0}^{T} \lambda^i \cdot S_{t-i}$$

where $\lambda \in (0, 1)$ is the decay factor. Typical values range from 0.9 (slow decay, multi-day impact) to 0.5 (fast decay, intraday impact). The optimal decay rate depends on the asset class and the type of news.

## Applications

### Earnings Call Sentiment

Earnings calls provide rich sentiment data. The tone of management during the Q&A session is particularly informative:

1. **Prepared remarks**: Usually positive and scripted; sentiment analysis here captures the degree of optimism.
2. **Q&A responses**: Unscripted; deviations from typical positive tone signal potential concerns.
3. **Analyst questions**: The sentiment of questions reveals market concerns and expectations.

Research shows that negative sentiment in earnings calls predicts negative abnormal returns over the following weeks.

### News Sentiment Trading

Financial news headlines and articles can be scored in real-time for sentiment-driven trading:

- **Breaking news**: Rapid sentiment scoring enables immediate position adjustments.
- **Sentiment reversals**: When sentiment shifts from negative to positive (or vice versa), it often signals a turning point.
- **Consensus sentiment**: When multiple sources agree on sentiment direction, the signal is stronger.

### Crypto Market Sentiment

Cryptocurrency markets are particularly sentiment-driven due to:

- Higher retail participation and social media influence
- Fewer fundamental anchors (no earnings, dividends, or book value)
- 24/7 trading with rapid information dissemination

FinBERT can be applied to crypto-specific text sources: Twitter/X discussions, Reddit posts, Telegram channels, and exchange announcements. The Bybit API provides the price data to correlate with sentiment signals.

## Rust Implementation

Our Rust implementation provides a complete sentiment analysis and trading toolkit with the following components:

### SentimentAnalyzer

The `SentimentAnalyzer` struct implements keyword-based financial sentiment analysis using the Loughran-McDonald financial sentiment dictionary. It scores text by counting positive and negative financial keywords, computing a normalized sentiment score between -1 and +1. This provides a fast, interpretable baseline that can process thousands of texts per second.

### SentimentAggregator

The `SentimentAggregator` collects individual sentiment scores over time and computes aggregated signals. It supports simple averaging, exponentially weighted averaging with configurable decay, and sentiment momentum (rate of change). These aggregated signals are more stable and predictive than individual text scores.

### SentimentClassifier

The `SentimentClassifier` implements logistic regression for binary sentiment classification (positive vs. negative). It accepts feature vectors derived from text analysis and trains using stochastic gradient descent. The classifier outputs both a sentiment direction and a confidence score.

### SentimentStrategy

The `SentimentStrategy` combines sentiment signals with price data to generate trading decisions. It uses configurable thresholds for entry and exit, supports both long and short positions, and tracks performance metrics including cumulative return, win rate, and Sharpe ratio.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint for backtesting sentiment strategies against real cryptocurrency price data.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain real-time market data for sentiment strategy backtesting:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used for correlating sentiment signals with price movements and computing strategy returns.
- **Order book endpoint** (`/v5/market/orderbook`): Provides a snapshot of the current limit order book. Used for computing spread and liquidity features that complement sentiment signals.

The Bybit API is well-suited for sentiment-based trading because it provides:
- Fine-grained intervals (1-minute klines for high-frequency sentiment trading)
- Multiple cryptocurrency pairs (BTCUSDT, ETHUSDT) to test cross-asset sentiment effects
- Consistent, low-latency responses suitable for real-time trading systems

## References

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. *arXiv preprint arXiv:1908.10063*.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171-4186.
3. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
4. Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782-796.
5. Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A large language model for extracting information from financial text. *Contemporary Accounting Research*, 40(2), 806-841.
