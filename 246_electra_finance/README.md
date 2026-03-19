# ELECTRA for Finance: Efficient Pre-training for Financial Text Analysis

ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) is a pre-training method for language models that replaces the Masked Language Modeling (MLM) objective with a more sample-efficient Replaced Token Detection (RTD) task. Instead of masking tokens and predicting them, ELECTRA trains a generator to replace tokens and a discriminator to detect which tokens were replaced — analogous to a Generative Adversarial Network (GAN) applied to text.

In finance, ELECTRA's efficiency advantage is particularly valuable:
- **Cost-effective training**: ELECTRA achieves BERT-level performance with significantly less compute, making it practical for financial institutions
- **Better sample efficiency**: Learns from ALL input tokens (not just masked ones), crucial when financial text data is limited
- **Smaller model sizes**: Competitive performance at smaller scales, enabling deployment on trading infrastructure
- **Domain adaptation**: Efficient fine-tuning on financial corpora (earnings calls, SEC filings, market news)

This chapter covers the theory and implementation of ELECTRA for financial NLP tasks, with practical applications using both stock market and cryptocurrency (Bybit) data.

## Content

1. [Introduction to ELECTRA](#introduction-to-electra)
2. [ELECTRA Architecture](#electra-architecture)
   * [Generator Network](#generator-network)
   * [Discriminator Network](#discriminator-network)
   * [Training Objective](#training-objective)
3. [ELECTRA vs BERT for Finance](#electra-vs-bert-for-finance)
4. [Financial Text Classification](#financial-text-classification)
   * [Sentiment Analysis](#sentiment-analysis)
   * [Event Detection](#event-detection)
5. [Implementation with PyTorch](#implementation-with-pytorch)
   * [Code Example: ELECTRA-based Sentiment Model](#code-example-electra-based-sentiment-model)
   * [Code Example: Trading Signal Generation](#code-example-trading-signal-generation)
6. [Trading Strategy Based on ELECTRA Signals](#trading-strategy-based-on-electra-signals)
7. [Backtesting the Strategy](#backtesting-the-strategy)
8. [Rust Implementation](#rust-implementation)
9. [References](#references)

## Introduction to ELECTRA

Traditional pre-training methods like BERT use Masked Language Modeling (MLM), where ~15% of tokens are masked and the model learns to predict them. This means the model only learns from a small fraction of input tokens per training step.

ELECTRA introduces a fundamentally different approach:

1. **Generator**: A small MLM model generates plausible replacement tokens
2. **Discriminator**: The main model classifies each token as "original" or "replaced"

```
Original:   "The stock [MASK] sharply after [MASK] report"
Generator:  "The stock  fell  sharply after  the  report"
                        ^^^^                 ^^^
Discriminator labels:  [orig, orig, replaced, orig, orig, orig, replaced, orig]
```

The discriminator learns from ALL tokens in every example, making training ~4x more efficient than BERT.

### Why ELECTRA Matters for Financial NLP

Financial text has unique characteristics that make ELECTRA particularly suitable:
- **Domain-specific vocabulary**: Financial terms, tickers, and jargon require efficient learning
- **Nuanced sentiment**: "Revenue grew 5%" can be positive or negative depending on expectations
- **Time sensitivity**: Models must be trained and updated quickly for market relevance
- **Limited labeled data**: Financial annotation is expensive and requires domain expertise

## ELECTRA Architecture

### Generator Network

The generator is a small transformer that performs masked language modeling. It takes corrupted input (with ~15% of tokens masked) and predicts the original tokens:

```
P_G(x_t | x_masked) = softmax(W_G · h_G(x_masked)_t)
```

The generator is typically 1/4 to 1/3 the size of the discriminator, providing a good balance between generating plausible replacements and keeping compute costs low.

### Discriminator Network

The discriminator is the main ELECTRA model. It receives the generator's output (with some tokens replaced) and must identify which tokens are original vs. replaced:

```
D(x_t) = sigmoid(w^T · h_D(x_replaced)_t)
```

For each position t, the discriminator outputs a probability that the token is from the original text rather than generated.

### Training Objective

The combined loss function balances generator and discriminator losses:

```
L = L_MLM(generator) + λ · L_Disc(discriminator)
```

Where:
- `L_MLM` is the standard masked language modeling loss for the generator
- `L_Disc` is the binary cross-entropy loss for the discriminator
- `λ` (typically 50) weights the discriminator loss higher since it's the primary model

After pre-training, the generator is discarded, and only the discriminator is used for downstream tasks.

## ELECTRA vs BERT for Finance

| Aspect | BERT | ELECTRA |
|--------|------|---------|
| Pre-training task | Masked LM (15% tokens) | Replaced Token Detection (100% tokens) |
| Training efficiency | Baseline | ~4x more efficient |
| Small model performance | Degrades significantly | Maintains competitive performance |
| Compute cost | High | Lower for same quality |
| Financial text adaptation | Good but expensive | Efficient domain adaptation |
| Inference speed | Same architecture | Same architecture |

For financial applications, ELECTRA's advantages compound:
- **Frequent retraining**: Markets evolve; ELECTRA can be retrained 4x faster
- **Multiple domains**: Equities, crypto, forex each need specialized models
- **Resource constraints**: Trading firms have compute budgets; ELECTRA does more with less

## Financial Text Classification

### Sentiment Analysis

ELECTRA excels at financial sentiment classification, where subtle language distinctions matter:

```python
# Financial sentiment examples
texts = [
    "Revenue exceeded analyst estimates by 12%",     # Positive
    "Revenue grew 3%, missing consensus by 200bps",  # Negative (missed expectations)
    "Company maintained its dividend guidance",       # Neutral
    "Bitcoin surged past key resistance at $45,000",  # Positive (crypto)
]
```

The model learns to distinguish between absolute performance ("revenue grew") and relative performance ("missed consensus"), a critical distinction in financial analysis.

### Event Detection

ELECTRA can classify financial events that drive trading decisions:

- **Earnings surprises**: Beat/miss/meet expectations
- **M&A activity**: Merger announcements, acquisition rumors
- **Regulatory actions**: SEC filings, policy changes
- **Market microstructure**: Exchange announcements, listing/delisting events

## Implementation with PyTorch

### Code Example: ELECTRA-based Sentiment Model

The `python/` directory contains a complete implementation:

```python
import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer

class FinancialElectra(nn.Module):
    """ELECTRA-based model for financial sentiment classification."""

    def __init__(self, num_classes=3, model_name='google/electra-small-discriminator',
                 dropout=0.3):
        super().__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        hidden_size = self.electra.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        return logits
```

### Code Example: Trading Signal Generation

```python
class ElectraTradingSignal:
    """Generate trading signals from financial text using ELECTRA."""

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def predict_sentiment(self, texts):
        """Predict sentiment for a batch of financial texts."""
        encodings = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings)
            probs = torch.softmax(logits, dim=-1)

        # Classes: 0=negative, 1=neutral, 2=positive
        return probs.cpu().numpy()

    def generate_signal(self, news_items, threshold=0.6):
        """Generate trading signal from multiple news items."""
        if not news_items:
            return 0.0, 0.0  # No signal, no confidence

        probs = self.predict_sentiment(news_items)
        avg_probs = probs.mean(axis=0)

        # Signal: positive - negative sentiment
        signal = avg_probs[2] - avg_probs[0]
        confidence = max(avg_probs[0], avg_probs[2])

        if confidence < threshold:
            return 0.0, confidence  # Not confident enough

        return signal, confidence
```

## Trading Strategy Based on ELECTRA Signals

The strategy combines ELECTRA sentiment signals with price data:

1. **News Collection**: Gather financial news and social media for target assets
2. **Sentiment Scoring**: Run each text through the ELECTRA model
3. **Signal Aggregation**: Combine multiple sentiment scores with time-decay weighting
4. **Position Sizing**: Scale positions based on sentiment confidence
5. **Risk Management**: Apply stop-losses and position limits

```python
def electra_trading_strategy(news_data, price_data, model, tokenizer,
                              signal_threshold=0.3, confidence_threshold=0.6):
    """Execute ELECTRA-based trading strategy."""
    signal_gen = ElectraTradingSignal(model, tokenizer)

    positions = []
    for date in price_data.index:
        # Get news for this date
        daily_news = news_data[news_data['date'] == date]['text'].tolist()

        signal, confidence = signal_gen.generate_signal(
            daily_news, threshold=confidence_threshold
        )

        if abs(signal) > signal_threshold:
            position_size = min(abs(signal) * confidence, 1.0)
            direction = 1 if signal > 0 else -1
            positions.append({
                'date': date,
                'direction': direction,
                'size': position_size,
                'signal': signal,
                'confidence': confidence
            })

    return pd.DataFrame(positions)
```

## Backtesting the Strategy

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted return: (Return - Rf) / Std |
| **Sortino Ratio** | Downside-adjusted return |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |

### Comparison Scenarios

We compare:
1. **Baseline**: Buy-and-hold strategy
2. **ELECTRA Sentiment**: Trade based on ELECTRA sentiment signals only
3. **ELECTRA + Technical**: Combine ELECTRA signals with technical indicators
4. **Ensemble**: Multiple ELECTRA models with confidence weighting

## Rust Implementation

The `rust_examples/` directory contains a production-ready Rust implementation with:

- **Bybit API integration**: Real-time cryptocurrency data fetching
- **Text preprocessing**: Tokenization and feature extraction in Rust
- **Sentiment scoring**: ONNX-based model inference for low latency
- **Backtesting engine**: High-performance strategy backtesting

### Running Rust Examples

```bash
cd rust_examples

# Fetch data from Bybit
cargo run --example fetch_data

# Run sentiment analysis and trading
cargo run --example electra_trading

# Run backtest
cargo run --example backtest
```

## References

1. **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**
   - Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020)
   - URL: https://arxiv.org/abs/2003.10555
   - *Introduced the ELECTRA pre-training method*

2. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models**
   - Araci, D. (2019)
   - URL: https://arxiv.org/abs/1908.10063
   - *Financial domain adaptation for transformer models*

3. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
   - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019)
   - URL: https://arxiv.org/abs/1810.04805
   - *Foundation model that ELECTRA improves upon*

4. **Attention Is All You Need**
   - Vaswani, A., et al. (2017)
   - URL: https://arxiv.org/abs/1706.03762
   - *Transformer architecture underlying ELECTRA*

5. **Natural Language Processing in Finance: A Survey**
   - Xing, F., Cambria, E., & Welsch, R. (2018)
   - URL: https://arxiv.org/abs/1807.02811
   - *Comprehensive survey of NLP methods for financial applications*
