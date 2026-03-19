# Chapter 250: GPT Financial Analysis

## Introduction

Generative Pre-trained Transformers (GPT) have emerged as a powerful paradigm for financial analysis, leveraging self-supervised learning on massive text corpora to develop rich representations of language, numbers, and context. Unlike traditional NLP models that require task-specific architectures, GPT models learn general-purpose representations during pre-training and can be adapted to diverse financial tasks — from earnings call summarization and sentiment analysis to numerical reasoning over financial statements and market commentary generation.

The core insight of GPT for finance is that financial text follows patterns at multiple levels: syntactic structure, domain-specific terminology, numerical relationships, and implicit sentiment. A sufficiently large language model pre-trained on financial corpora can internalize these patterns and then apply them to downstream tasks with minimal fine-tuning. This "pre-train then adapt" approach has proven particularly effective in finance, where labeled data is scarce but raw text (SEC filings, analyst reports, news articles, earnings transcripts) is abundant.

This chapter presents a framework for applying GPT-style models to financial analysis. We cover the transformer architecture underpinning GPT, the key adaptation strategies (prompt engineering, fine-tuning, few-shot learning), and a working Rust implementation that performs sentiment-driven trading signal generation using data from both stock markets and the Bybit cryptocurrency exchange.

## Key Concepts

### The Transformer Architecture

The GPT family is built on the transformer decoder architecture introduced by Vaswani et al. (2017). The key innovation is the self-attention mechanism, which allows each token in a sequence to attend to all previous tokens, capturing long-range dependencies without the sequential bottleneck of RNNs.

Given an input sequence of token embeddings $\mathbf{X} \in \mathbb{R}^{T \times d}$, self-attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where the queries, keys, and values are linear projections of the input:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

Multi-head attention extends this by running $h$ parallel attention heads with different learned projections, enabling the model to capture different types of relationships simultaneously:

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

### Causal Language Modeling

GPT models are trained with a causal (autoregressive) language modeling objective. Given a sequence of tokens $(x_1, x_2, \ldots, x_T)$, the model maximizes the log-likelihood:

$$\mathcal{L} = \sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)$$

This is enforced by a causal mask in the attention computation that prevents each position from attending to future positions. The model learns to predict the next token given all preceding context, which implicitly requires understanding syntax, semantics, and factual relationships.

For financial text, this objective teaches the model patterns like: after "Revenue increased by 15% to $", the next tokens are likely a dollar amount consistent with 15% growth from the previous period's revenue.

### Financial Sentiment Analysis

Sentiment analysis in finance goes beyond simple positive/negative classification. Financial sentiment is nuanced and context-dependent:

- **Bullish/Bearish signals**: Direct indicators of market direction expectations
- **Uncertainty quantification**: Language indicating confidence or doubt about forecasts
- **Forward-looking statements**: Distinguishing past performance from future guidance
- **Numerical sentiment**: "Revenue of $1.2B" is positive if consensus was $1.1B, negative if it was $1.3B

A GPT model fine-tuned on financial text can capture these nuances because it has learned the distributional patterns of financial language during pre-training. The sentiment score for a document $d$ can be expressed as:

$$S(d) = \sigma\left(\mathbf{w}^T \mathbf{h}_{\text{[CLS]}} + b\right)$$

where $\mathbf{h}_{\text{[CLS]}}$ is the hidden state of the classification token from the final transformer layer, and $\sigma$ is the sigmoid function mapping to $[0, 1]$.

### Prompt Engineering for Financial Tasks

Rather than fine-tuning, GPT models can be steered via carefully crafted prompts. This is particularly useful in finance where:

1. **Zero-shot analysis**: "Classify the following earnings report excerpt as bullish, neutral, or bearish: [text]"
2. **Few-shot learning**: Providing 2-5 labeled examples before the query teaches the model the task format and expected output
3. **Chain-of-thought reasoning**: "Analyze the following financial statement step by step, then provide a trading recommendation: [text]"

The effectiveness of prompt engineering depends on the prompt template $\mathcal{T}$, the verbalizer $\mathcal{V}$ mapping label words to classes, and the number of demonstrations $k$:

$$P(y | x) = P(\mathcal{V}(y) | \mathcal{T}(x, \{(x_i, y_i)\}_{i=1}^{k}))$$

### Numerical Reasoning in Financial Context

A key challenge for GPT in finance is numerical reasoning. Financial analysis requires:

- **Percentage calculations**: Computing growth rates, margins, ratios
- **Comparative reasoning**: Determining if a metric exceeds expectations
- **Temporal reasoning**: Understanding year-over-year and quarter-over-quarter changes
- **Scale awareness**: Distinguishing millions from billions, basis points from percentages

Recent research shows that GPT models can perform basic numerical reasoning when numbers are tokenized appropriately and the model has seen sufficient examples of financial calculations during pre-training.

## ML Approaches

### Fine-Tuned GPT for Sentiment Classification

The most direct application fine-tunes a pre-trained GPT model on labeled financial sentiment data. The process involves:

1. **Pre-processing**: Tokenize financial texts, handling domain-specific tokens (ticker symbols, financial abbreviations, numerical formats)
2. **Supervised fine-tuning**: Train on labeled examples with cross-entropy loss:

$$\mathcal{L}_{\text{FT}} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

3. **Regularization**: Apply dropout, weight decay, and early stopping to prevent overfitting on small financial datasets

The fine-tuned model produces a probability distribution over sentiment classes (bullish, neutral, bearish) for each input text.

### GPT-Based Feature Extraction for Trading

Instead of using GPT's output directly, we can extract embedding features and feed them into traditional trading models:

1. Extract the hidden state $\mathbf{h} \in \mathbb{R}^d$ from the last transformer layer
2. Combine with numerical features (price, volume, technical indicators) to form $\mathbf{x} = [\mathbf{h}; \mathbf{f}_{\text{numerical}}]$
3. Train a gradient boosting or neural network model on the combined features:

$$\hat{y} = f_{\text{trading}}(\mathbf{x}) = f_{\text{trading}}([\mathbf{h}; \mathbf{f}_{\text{numerical}}])$$

This approach leverages GPT's language understanding while allowing specialized trading models to handle the prediction.

### Ensemble with Technical Analysis

The strongest approach combines GPT sentiment signals with traditional technical and order flow features:

$$\text{Signal}_t = \alpha \cdot S_t^{\text{GPT}} + \beta \cdot S_t^{\text{technical}} + \gamma \cdot S_t^{\text{flow}}$$

where $S_t^{\text{GPT}}$ is the GPT-derived sentiment score, $S_t^{\text{technical}}$ aggregates moving averages, RSI, and MACD, and $S_t^{\text{flow}}$ captures order flow imbalance. The weights $\alpha, \beta, \gamma$ are learned via cross-validation on historical data.

## Feature Engineering

### Text Features from Financial Documents

Key text features that GPT can extract or enhance:

- **Entity-level sentiment**: Sentiment about specific companies, sectors, or products mentioned in the text
- **Event detection**: Identifying material events (mergers, earnings surprises, regulatory actions)
- **Topic distribution**: The mix of topics discussed (revenue, margins, guidance, macro outlook)
- **Linguistic complexity**: Readability and obfuscation metrics that correlate with negative information hiding

### Numerical Feature Integration

Financial GPT analysis benefits from combining text features with structured data:

- **Price momentum**: Returns over 1, 5, 20, 60 day windows
- **Volatility regime**: Rolling standard deviation and implied volatility levels
- **Volume profile**: Volume relative to moving average, indicating unusual activity
- **Market microstructure**: Bid-ask spread, depth, and order flow metrics from exchange data

## Applications

### Earnings Call Analysis

GPT models excel at analyzing earnings call transcripts, which contain both prepared remarks and spontaneous Q&A responses. The model can:

- Detect changes in management tone compared to previous quarters
- Identify hedging language and qualifiers that signal uncertainty
- Extract forward guidance and compare it to consensus expectations
- Score the overall bullish/bearish lean of the call

### News-Based Trading Signals

Real-time news analysis is one of the highest-value applications. The GPT model processes news articles and produces:

- **Relevance score**: Is this news material for a given asset?
- **Sentiment score**: Bullish or bearish implications
- **Novelty score**: Is this new information or a rehash of known facts?
- **Urgency score**: How quickly should a trader react?

### Risk Report Generation

GPT can generate structured risk reports from unstructured data, summarizing:

- Key risk factors identified across multiple filings
- Changes in risk language over time
- Emerging risks not previously disclosed
- Correlation between risk language and subsequent market moves

## Rust Implementation

### Architecture Overview

Our Rust implementation provides a lightweight GPT-inspired framework for financial text analysis. Rather than implementing a full transformer (which requires GPU support), we implement a simplified attention-based text analysis pipeline that demonstrates the key concepts:

1. **Tokenizer**: Splits financial text into tokens with domain-specific vocabulary
2. **Embedding Layer**: Maps tokens to dense vectors using pre-computed financial embeddings
3. **Attention Mechanism**: Single-head scaled dot-product attention for feature extraction
4. **Sentiment Classifier**: Linear classifier on attended representations
5. **Trading Signal Generator**: Combines text sentiment with price/volume features

### Token Embedding

The `FinancialTokenizer` struct handles text preprocessing with awareness of financial terminology:

```rust
pub struct FinancialTokenizer {
    vocab: HashMap<String, usize>,
    embeddings: Vec<Vec<f64>>,
}
```

It recognizes ticker symbols, numerical patterns, and financial keywords, mapping each to a learned embedding vector.

### Attention Computation

The `AttentionLayer` implements scaled dot-product attention:

```rust
pub struct AttentionLayer {
    w_query: Vec<Vec<f64>>,
    w_key: Vec<Vec<f64>>,
    w_value: Vec<Vec<f64>>,
    d_k: f64,
}
```

This allows the model to weight different parts of the input text based on their relevance to the financial analysis task.

### Sentiment Classification

The `SentimentClassifier` produces a three-class output (bullish, neutral, bearish) using a softmax layer:

```rust
pub struct SentimentClassifier {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
}
```

### Trading Signal Integration

The `TradingSignalGenerator` combines GPT sentiment with technical features:

```rust
pub struct TradingSignalGenerator {
    sentiment_weight: f64,
    momentum_weight: f64,
    volatility_weight: f64,
    threshold: f64,
}
```

### Bybit API Integration

The `BybitClient` struct connects to the Bybit V5 API to fetch real-time market data:

```rust
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}
```

It retrieves kline (candlestick) data and order book snapshots, providing the numerical context that complements the text analysis.

## References

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.
2. Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
3. Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS, 2020.
4. Yang, Y., Uy, M.C.S., and Huang, A. "FinBERT: A Pretrained Language Model for Financial Communications." arXiv:2006.08097, 2020.
5. Lopez-Lira, A. and Tang, Y. "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models." arXiv:2304.07619, 2023.
6. Wu, S., et al. "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564, 2023.
7. Loughran, T. and McDonald, B. "When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks." Journal of Finance, 2011.

## Running the Code

```bash
cd rust
cargo test
cargo run --example trading_example
```
