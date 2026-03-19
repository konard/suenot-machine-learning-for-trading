# Chapter 286: T5 Pretraining for Finance

## 1. Introduction

The **T5 (Text-to-Text Transfer Transformer)** model, introduced by Raffel et al. (2020), represents a paradigm shift in how we approach natural language processing tasks. Unlike earlier architectures that required task-specific heads and output layers, T5 casts every NLP problem into a unified text-to-text format: the model always takes text as input and produces text as output. Whether the task is classification, translation, summarization, or question answering, T5 treats them all identically.

For financial applications, this unification is remarkably powerful. Consider the diversity of text-based tasks a quantitative trader encounters daily:

- **Sentiment analysis** of earnings calls, SEC filings, and analyst reports
- **Summarization** of lengthy financial documents into actionable intelligence
- **Question answering** over structured and unstructured financial data
- **Named entity recognition** for extracting tickers, executives, and financial metrics
- **Text classification** for categorizing news by sector, event type, or relevance

With T5, all of these tasks share a single model architecture and training pipeline. By pretraining on a large financial corpus and then fine-tuning with task-specific prefixes (e.g., "classify sentiment:", "summarize:", "answer:"), we obtain a versatile financial NLP engine that can power a multi-signal trading system.

In this chapter, we explore the mathematical foundations of T5, discuss how to adapt its pretraining and fine-tuning procedures for financial text, and implement a simplified T5-inspired encoder-decoder model in Rust. We then integrate this model with Bybit market data to create a sentiment-driven trading strategy.

## 2. Mathematical Foundations

### 2.1 Encoder-Decoder Architecture

T5 uses the original Transformer encoder-decoder architecture. Given an input sequence $X = (x_1, x_2, \ldots, x_n)$, the encoder produces contextualized representations $H = (h_1, h_2, \ldots, h_n)$, and the decoder generates the output sequence $Y = (y_1, y_2, \ldots, y_m)$ autoregressively.

**Encoder Layer**: Each encoder layer applies multi-head self-attention followed by a position-wise feed-forward network (FFN):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

For multi-head attention with $h$ heads:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)
$$

The feed-forward network applies two linear transformations with a ReLU activation:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

Layer normalization and residual connections wrap each sub-layer:

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

**Decoder Layer**: The decoder has three sub-layers: (1) masked self-attention over previously generated tokens, (2) cross-attention over the encoder output $H$, and (3) a feed-forward network.

The cross-attention mechanism uses decoder states as queries and encoder outputs as keys and values:

$$
\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}
$$

### 2.2 Span Corruption Pretraining

T5 uses a denoising objective called **span corruption**. During pretraining, contiguous spans of tokens are replaced with unique sentinel tokens, and the model is trained to reconstruct the original spans.

Given input text: "The stock market rallied after the Federal Reserve announced rate cuts"

After span corruption (15% of tokens masked in contiguous spans):

- **Input**: "The stock `<extra_id_0>` after the `<extra_id_1>` rate cuts"
- **Target**: "`<extra_id_0>` market rallied `<extra_id_1>` Federal Reserve announced"

The training objective minimizes the negative log-likelihood:

$$
\mathcal{L} = -\sum_{t=1}^{m} \log P(y_t | y_{<t}, X_{\text{corrupted}}; \theta)
$$

The span corruption approach is superior to individual token masking (as in BERT) because it forces the model to learn longer-range dependencies and reconstruct coherent phrases.

### 2.3 Task Unification as Text-to-Text

The key innovation of T5 is unifying all tasks into a text-to-text framework using task prefixes:

| Task | Input | Output |
|------|-------|--------|
| Sentiment | "classify sentiment: Revenue beat expectations by 20%" | "positive" |
| Summarization | "summarize: [long earnings call transcript]" | "[concise summary]" |
| QA | "answer: What was the EPS? context: The company reported EPS of $2.15" | "$2.15" |
| Translation | "translate English to signals: Markets crashed on recession fears" | "sell" |

The loss function remains the same for all tasks:

$$
\mathcal{L}_{\text{task}} = -\sum_{(x,y) \in \mathcal{D}_{\text{task}}} \log P(y | \text{prefix} \oplus x; \theta)
$$

where $\oplus$ denotes concatenation and $\text{prefix}$ is a task-specific string.

## 3. Financial Fine-Tuning

### 3.1 Sentiment Classification

Financial sentiment differs from general sentiment. The phrase "the company burned through $500M in cash" carries negative sentiment, while "the company burned through expectations" might be positive. Fine-tuning T5 on financial corpora (e.g., Financial PhraseBank, SEC filings, earnings transcripts) captures these domain-specific nuances.

The fine-tuning setup:
- **Input**: `"classify financial sentiment: <financial text>"`
- **Output**: `"positive"`, `"negative"`, or `"neutral"`

These labels map directly to trading signals:
- **positive** -> BUY signal (bullish sentiment)
- **negative** -> SELL signal (bearish sentiment)
- **neutral** -> HOLD signal (no action)

### 3.2 Summarization of Financial Documents

Earnings calls and SEC filings are notoriously long. T5 can summarize these into concise, actionable paragraphs:

- **Input**: `"summarize: [10-K filing section, ~5000 words]"`
- **Output**: `"Revenue grew 15% YoY driven by cloud services. Operating margins expanded 200bps. Management guided for continued growth in FY2025."`

### 3.3 Question Answering on Earnings Calls

Analysts need to quickly extract specific information from transcripts:

- **Input**: `"answer: What was guidance for next quarter? context: [earnings call transcript]"`
- **Output**: `"Management expects revenue of $4.2-4.5 billion with margins of 22-24%"`

### 3.4 Multi-Task Training

A key advantage of the T5 framework is **multi-task training**. By combining datasets for sentiment, summarization, and QA, we train a single model that excels across all financial NLP tasks. The multi-task loss is:

$$
\mathcal{L}_{\text{multi}} = \sum_{k} \lambda_k \mathcal{L}_k
$$

where $\lambda_k$ are task-specific weights that can be tuned based on downstream utility.

## 4. Rust Implementation

Our Rust implementation provides a simplified T5-inspired encoder-decoder architecture. The key components are:

### 4.1 Architecture Overview

```
Input Text -> Tokenization -> Encoder (Self-Attention + FFN)
                                  |
                                  v
                            Encoder Output
                                  |
                                  v
                     Decoder (Cross-Attention + Self-Attention + FFN)
                                  |
                                  v
                     Sentiment Classification Head -> BUY/SELL/HOLD
```

### 4.2 Key Components

1. **T5Config**: Configuration struct holding model hyperparameters (d_model, num_heads, num_layers, d_ff, vocab_size)
2. **MultiHeadAttention**: Implements scaled dot-product attention with multiple heads
3. **FeedForward**: Position-wise feed-forward network with ReLU activation
4. **EncoderLayer**: Self-attention + FFN with residual connections
5. **DecoderLayer**: Self-attention + cross-attention + FFN
6. **T5Model**: Full encoder-decoder model with sentiment classification head
7. **SpanCorruptor**: Implements the span corruption pretraining objective
8. **BybitClient**: Fetches real-time market data from Bybit API

### 4.3 Implementation Details

The implementation uses `ndarray` for tensor operations and provides:

- Forward pass through encoder and decoder stacks
- Span corruption for pretraining data generation
- Sentiment-to-signal mapping for trading decisions
- Integration with Bybit API for live market data

See `rust/src/lib.rs` for the complete implementation and `rust/examples/trading_example.rs` for the trading strategy.

## 5. Bybit Data Integration

### 5.1 Data Pipeline

The integration with Bybit follows this pipeline:

1. **Fetch OHLCV data** from Bybit API for a target symbol (e.g., BTCUSDT)
2. **Generate simulated news headlines** (in production, these would come from a news API)
3. **Run T5 sentiment analysis** on each headline
4. **Map sentiment to trading signals** (positive -> buy, negative -> sell, neutral -> hold)
5. **Execute backtest** combining price data with sentiment signals

### 5.2 Signal Generation

The T5 model outputs sentiment probabilities which are converted to signals:

```
Sentiment Score > 0.6  -> BUY signal  (confidence: score)
Sentiment Score < 0.4  -> SELL signal (confidence: 1 - score)
Otherwise              -> HOLD        (no position change)
```

### 5.3 Risk Management

Sentiment-based strategies require careful risk management:

- **Position sizing**: Scale position size by sentiment confidence
- **Stop losses**: Fixed percentage stops independent of sentiment
- **Signal decay**: Reduce signal strength as time passes since the news event
- **Confirmation**: Require multiple corroborating signals before large positions

## 6. Key Takeaways

1. **T5's text-to-text paradigm** unifies all financial NLP tasks under a single framework, simplifying model management and deployment in production trading systems.

2. **Span corruption pretraining** on financial corpora teaches the model to understand financial language, jargon, and domain-specific relationships before task-specific fine-tuning.

3. **Multi-task fine-tuning** allows a single T5 model to handle sentiment analysis, summarization, and question answering simultaneously, reducing infrastructure complexity.

4. **Financial sentiment mapping** requires domain expertise -- the same words can carry opposite sentiment in financial vs. general contexts.

5. **Encoder-decoder architecture** is essential for generative tasks like summarization and QA, giving T5 an advantage over encoder-only models (BERT) for these applications.

6. **Bybit integration** demonstrates how NLP signals can be combined with real-time market data for algorithmic trading, creating a bridge between unstructured text and quantitative signals.

7. **Risk management** remains critical: sentiment signals are noisy and should be combined with technical indicators and proper position sizing for robust trading strategies.

8. **Rust implementation** provides the performance characteristics needed for real-time NLP inference in latency-sensitive trading environments, while maintaining memory safety guarantees.

## References

- Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR.
- Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
- Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models."
- Yang, Y. et al. (2020). "FinBERT: A Pretrained Language Model for Financial Communications."
- Liu, Z. et al. (2021). "FinQA: A Dataset of Numerical Reasoning over Financial Data."
