# Chapter 244: XLNet for Finance

## Introduction

XLNet is a generalized autoregressive pretraining method that combines the strengths of autoregressive (AR) language models and autoencoding (AE) approaches like BERT. Introduced by Yang et al. (2019), XLNet addresses key limitations of BERT by using a permutation-based training objective that captures bidirectional context without relying on masked tokens and the independence assumption they introduce.

In financial applications, XLNet's permutation language modeling is particularly valuable. Financial texts — earnings calls, analyst reports, regulatory filings, and news articles — contain complex dependencies between entities, numbers, and sentiments. BERT's masked language model assumes that masked tokens are independent of each other given the unmasked tokens, which can miss important inter-token dependencies. XLNet's permutation objective captures these dependencies naturally, making it better suited for understanding nuanced financial language where the relationship between a company name, a financial metric, and a sentiment word matters deeply.

This chapter covers the theoretical foundations of XLNet, its advantages over BERT for financial NLP, and provides working implementations in both Python and Rust. We demonstrate sentiment analysis on financial texts and build a trading signal generator that connects to the Bybit cryptocurrency exchange.

## Key Concepts

### Autoregressive vs. Autoencoding Language Models

**Autoregressive (AR) models** like GPT estimate the probability of a text sequence by factoring the joint probability into a product of conditional probabilities in a fixed left-to-right order:

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t | x_1, \ldots, x_{t-1})$$

AR models are strong at generation but can only capture context in one direction.

**Autoencoding (AE) models** like BERT use a denoising objective. Given a corrupted input $\hat{\mathbf{x}}$ (with some tokens replaced by `[MASK]`), the model reconstructs the original tokens:

$$\max_\theta \log p(\bar{\mathbf{x}} | \hat{\mathbf{x}}) \approx \sum_{t=1}^{T} m_t \log p(x_t | \hat{\mathbf{x}})$$

where $m_t = 1$ indicates token $t$ was masked. The key limitation is the **independence assumption**: BERT assumes masked tokens are independent of each other given the unmasked context, which ignores correlations between masked positions.

### Permutation Language Modeling

XLNet's central innovation is **permutation language modeling**. Instead of fixing the factorization order (left-to-right) or masking tokens (BERT), XLNet considers all possible permutations of the factorization order and maximizes the expected log-likelihood:

$$\max_\theta \; \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_T} \left[ \sum_{t=1}^{T} \log p_\theta(x_{z_t} | \mathbf{x}_{\mathbf{z}_{<t}}) \right]$$

where $\mathcal{Z}_T$ is the set of all permutations of $\{1, 2, \ldots, T\}$, and $\mathbf{z}$ is a specific permutation. For each permutation $\mathbf{z}$, token $x_{z_t}$ is predicted using only the tokens that appear before it in the permutation order $\mathbf{x}_{\mathbf{z}_{<t}}$.

This achieves bidirectional context (every token can attend to tokens on both sides in the original sequence) while maintaining an autoregressive factorization (no independence assumption between predicted tokens).

### Two-Stream Self-Attention

Standard Transformers cannot implement permutation language modeling because the attention mechanism needs to know *which* position is being predicted (to avoid trivially looking up the answer). XLNet solves this with **two-stream self-attention**:

1. **Content stream** $h_{z_t}$: A standard hidden state that encodes the content of token $x_{z_t}$ along with its context. It can attend to all tokens at positions $z_{\leq t}$ (including itself).

2. **Query stream** $g_{z_t}$: A separate hidden state that encodes only the position $z_t$ and the context $\mathbf{x}_{\mathbf{z}_{<t}}$ (excluding the token at $z_t$ itself). It is used to predict the token at position $z_t$.

The update rules for layer $m$ are:

$$g_{z_t}^{(m)} \leftarrow \text{Attention}(Q = g_{z_t}^{(m-1)}, \; KV = h_{\mathbf{z}_{<t}}^{(m-1)}; \theta)$$

$$h_{z_t}^{(m)} \leftarrow \text{Attention}(Q = h_{z_t}^{(m-1)}, \; KV = h_{\mathbf{z}_{\leq t}}^{(m-1)}; \theta)$$

### Transformer-XL Backbone

XLNet builds on Transformer-XL, which introduces two mechanisms for handling long sequences:

**Segment-level recurrence**: Hidden states from previous segments are cached and reused as extended context for the current segment:

$$\tilde{h}^{(n-1)} = [\text{SG}(h_{\tau-1}^{(n-1)}) \circ h_{\tau}^{(n-1)}]$$

where $\text{SG}(\cdot)$ denotes stop-gradient and $\circ$ denotes concatenation along the sequence dimension. This allows the model to capture dependencies beyond the segment length without backpropagating through previous segments.

**Relative positional encoding**: Instead of absolute position embeddings, Transformer-XL uses relative position encodings that depend on the distance between tokens:

$$A_{i,j} = E_{x_i}^T W_q^T W_{k,E} E_{x_j} + E_{x_i}^T W_q^T W_{k,R} R_{i-j} + u^T W_{k,E} E_{x_j} + v^T W_{k,R} R_{i-j}$$

This enables the model to generalize to sequences longer than those seen during training, which is critical for processing lengthy financial documents.

## Financial NLP Applications

### Sentiment Analysis

Financial sentiment analysis with XLNet differs from general sentiment analysis in several ways:

- **Domain-specific vocabulary**: Words like "liability," "short," and "correction" have different connotations in finance than in general English.
- **Numerical reasoning**: Financial texts contain numbers whose magnitude and context determine sentiment. "Revenue grew 2%" vs. "Revenue grew 200%" carry very different signals.
- **Negation and hedging**: Financial language frequently uses hedging ("may," "could," "subject to") and negation, which models must handle carefully.
- **Forward-looking statements**: Earnings calls and filings contain forward-looking language that requires understanding temporal context.

XLNet's permutation objective naturally captures long-range dependencies between these elements. For example, in the sentence "Despite strong revenue growth, the company's guidance for next quarter was unexpectedly weak," the sentiment depends on the interaction between "strong revenue growth" (positive) and "unexpectedly weak guidance" (negative), with the latter dominating.

### Document Classification

Financial documents can be classified into categories such as:
- **Earnings reports**: Positive outlook, negative outlook, neutral
- **News articles**: Bullish, bearish, informational
- **SEC filings**: Risk factors, material events, routine updates

XLNet's ability to handle long documents (via Transformer-XL's recurrence mechanism) makes it particularly suited for classifying SEC filings and earnings transcripts that can span thousands of tokens.

### Named Entity Recognition in Finance

Financial NER identifies entities such as company names, ticker symbols, financial metrics, monetary amounts, dates, and regulatory bodies. XLNet's bidirectional context understanding helps disambiguate entities — for example, distinguishing "Apple" the company from "apple" the fruit based on surrounding financial context.

## Trading Signal Generation

### Sentiment-Based Signals

The pipeline for generating trading signals from XLNet sentiment analysis:

1. **Data collection**: Gather financial texts (news, social media, filings) for target assets.
2. **Preprocessing**: Tokenize using SentencePiece (XLNet's tokenizer), handle financial abbreviations and ticker symbols.
3. **Sentiment scoring**: Run texts through fine-tuned XLNet to produce sentiment scores in $[-1, 1]$.
4. **Signal aggregation**: Aggregate scores over a time window using exponential decay:

$$S_t = \sum_{i=0}^{N} \alpha^i \cdot s_{t-i}$$

where $s_{t-i}$ is the sentiment score of a text published at time $t-i$ and $\alpha \in (0, 1)$ is the decay factor.

5. **Position sizing**: Map the aggregated signal to a position size using a sigmoid function:

$$\text{position}_t = 2 \cdot \sigma(\beta \cdot S_t) - 1$$

where $\beta$ controls the aggressiveness of the strategy.

### Combining with Price Data

Sentiment signals are most effective when combined with price-based features:

- **Sentiment-momentum alignment**: When sentiment and price momentum agree, the signal is stronger.
- **Sentiment divergence**: When sentiment turns negative but price continues rising, this is a contrarian warning signal.
- **Volatility adjustment**: Scale position sizes inversely with recent volatility to maintain consistent risk.

## Python Implementation

### Sentiment Analysis with Hugging Face

```python
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch
import numpy as np

class FinancialSentimentAnalyzer:
    """XLNet-based sentiment analyzer for financial texts."""

    def __init__(self, model_name="xlnet-base-cased", num_labels=3):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.eval()
        self.label_map = {0: "bearish", 1: "neutral", 2: "bullish"}

    def analyze(self, text, max_length=512):
        """Analyze sentiment of a financial text.

        Returns a dict with label and confidence scores.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        predicted_label = int(np.argmax(probs))
        return {
            "label": self.label_map[predicted_label],
            "confidence": float(probs[predicted_label]),
            "scores": {
                self.label_map[i]: float(probs[i]) for i in range(len(probs))
            },
        }

    def analyze_batch(self, texts, max_length=512):
        """Analyze sentiment for a batch of texts."""
        return [self.analyze(text, max_length) for text in texts]
```

### Trading Signal Generator

```python
import math

class SentimentSignalGenerator:
    """Generate trading signals from sentiment scores."""

    def __init__(self, decay=0.9, aggressiveness=2.0):
        self.decay = decay
        self.aggressiveness = aggressiveness
        self.history = []

    def update(self, sentiment_score):
        """Add a new sentiment score and compute the aggregated signal."""
        self.history.append(sentiment_score)
        signal = sum(
            self.decay ** i * self.history[-(i + 1)]
            for i in range(len(self.history))
        )
        return signal

    def position(self, signal):
        """Convert aggregated signal to position size in [-1, 1]."""
        return 2.0 * (1.0 / (1.0 + math.exp(-self.aggressiveness * signal))) - 1.0

    def generate(self, sentiment_score):
        """Update and return position recommendation."""
        signal = self.update(sentiment_score)
        pos = self.position(signal)
        if pos > 0.3:
            action = "BUY"
        elif pos < -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        return {"signal": signal, "position": pos, "action": action}
```

### Backtesting Framework

```python
class SentimentBacktester:
    """Backtest a sentiment-based trading strategy."""

    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0
        self.trades = []
        self.equity_curve = [initial_capital]

    def step(self, price, sentiment_position):
        """Execute one step of the backtest.

        Args:
            price: Current asset price.
            sentiment_position: Target position from signal generator [-1, 1].
        """
        target_units = sentiment_position * self.capital / price
        delta = target_units - self.position
        if abs(delta) > 0.001:
            self.trades.append({
                "price": price,
                "units": delta,
                "direction": "buy" if delta > 0 else "sell",
            })
        self.position = target_units
        equity = self.capital + self.position * price
        self.equity_curve.append(equity)
        return equity

    def metrics(self):
        """Compute performance metrics."""
        returns = [
            (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
            for i in range(1, len(self.equity_curve))
            if self.equity_curve[i - 1] != 0
        ]
        if not returns:
            return {}
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0.0
        max_equity = self.equity_curve[0]
        max_drawdown = 0.0
        for eq in self.equity_curve:
            max_equity = max(max_equity, eq)
            drawdown = (max_equity - eq) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
        downside = [r for r in returns if r < 0]
        downside_std = (
            (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
            if downside else 0.0
        )
        sortino = (avg_return / downside_std * (252 ** 0.5)) if downside_std > 0 else 0.0
        return {
            "total_return": (self.equity_curve[-1] - self.initial_capital) / self.initial_capital,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "num_trades": len(self.trades),
        }
```

## Rust Implementation

The Rust implementation provides a high-performance sentiment scoring engine and trading signal generator with Bybit API integration. See the `rust/` directory for the complete source code.

### Core Components

- **`SentimentScorer`**: A logistic regression-based sentiment classifier that scores financial texts using TF-IDF-like features. Supports both single-text and batch inference.
- **`SignalGenerator`**: Converts sentiment scores into trading signals using exponential decay aggregation and sigmoid-based position sizing.
- **`Backtester`**: Simulates a sentiment-driven trading strategy on historical price data and computes performance metrics (Sharpe ratio, Sortino ratio, max drawdown).
- **`BybitClient`**: Async HTTP client for Bybit V5 API. Fetches kline data for backtesting and live signal generation.

### Usage

```bash
cd rust
cargo build
cargo run --example trading_example
```

## Bybit API Integration

The implementation fetches cryptocurrency market data from Bybit's V5 REST API:

- **Kline endpoint** (`/v5/market/kline`): Retrieves OHLCV candlestick data at configurable intervals. Used for backtesting sentiment-based strategies on crypto pairs like BTCUSDT and ETHUSDT.

The Bybit API provides:
- Multiple timeframe support (1m, 5m, 15m, 1h, 4h, 1d)
- Historical data for backtesting
- Low-latency responses for live trading integration

## Comparison with BERT

| Feature | BERT | XLNet |
|---------|------|-------|
| Training objective | Masked Language Model | Permutation Language Model |
| Independence assumption | Yes (masked tokens assumed independent) | No (autoregressive factorization) |
| Long-range dependencies | Limited by fixed context window | Extended via Transformer-XL recurrence |
| Positional encoding | Absolute | Relative |
| Pre-training data efficiency | Lower (only 15% tokens predicted) | Higher (all tokens predicted) |
| Financial text performance | Strong baseline | Better on long documents and complex dependencies |

## References

1. Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le, Q. V. (2019). XLNet: Generalized Autoregressive Pretraining for Language Understanding. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1906.08237
2. Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*.
3. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv preprint arXiv:1908.10063*.
4. Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782-796.
5. Loughran, T., & McDonald, B. (2011). When is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
