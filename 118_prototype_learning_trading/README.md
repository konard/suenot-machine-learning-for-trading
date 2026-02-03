# Prototype Learning for Market State Classification

Prototype-based learning provides an interpretable approach to understanding market conditions by learning representative patterns (prototypes) that characterize different market states. This chapter implements the "This Looks Like That" (ProtoPNet) framework, adapted for financial time series analysis and trading strategy development.

Unlike black-box models, prototype learning offers transparent reasoning: it classifies market conditions by finding similarity to learned prototypes, enabling traders to understand *why* a particular prediction was made. This interpretability is crucial in financial applications where understanding model behavior is as important as prediction accuracy.

## Content

1. [Introduction to Prototype Learning](#introduction-to-prototype-learning)
2. [Mathematical Framework](#mathematical-framework)
    * [Prototype Networks Architecture](#prototype-networks-architecture)
    * [Similarity Functions](#similarity-functions)
    * [Training Objective](#training-objective)
3. [Market State Classification](#market-state-classification)
    * [Defining Market States](#defining-market-states)
    * [Feature Engineering for Prototypes](#feature-engineering-for-prototypes)
4. [Implementation](#implementation)
    * [Code Example: Python Implementation](#code-example-python-implementation)
    * [Code Example: Rust Implementation](#code-example-rust-implementation)
5. [Trading Strategy](#trading-strategy)
    * [Signal Generation](#signal-generation)
    * [Backtesting Framework](#backtesting-framework)
6. [Data Sources](#data-sources)
    * [Stock Market Data](#stock-market-data)
    * [Cryptocurrency Data (Bybit)](#cryptocurrency-data-bybit)
7. [Evaluation Metrics](#evaluation-metrics)
8. [References](#references)

## Introduction to Prototype Learning

Prototype learning is a form of case-based reasoning where classification decisions are made by comparing new instances to representative examples (prototypes) learned during training. The key insight from the ProtoPNet paper is that these prototypes can be learned end-to-end alongside the classification model.

In the context of trading:
- **Prototypes represent typical market patterns** - e.g., consolidation before breakout, trending momentum, mean-reversion setups
- **Classification is interpretable** - "This market looks like [learned prototype for bullish continuation]"
- **Reasoning is transparent** - Traders can inspect what patterns the model has learned

### Why Prototype Learning for Trading?

1. **Interpretability**: Understand why the model predicts a particular market state
2. **Regulatory Compliance**: Explain model decisions to regulators and stakeholders
3. **Risk Management**: Identify when market conditions don't match any known prototype
4. **Strategy Refinement**: Learn new market patterns from data that complement human intuition

## Mathematical Framework

### Prototype Networks Architecture

The prototype network consists of three main components:

1. **Feature Encoder** $f: \mathbb{R}^{T \times D} \rightarrow \mathbb{R}^{H}$
   - Maps input time series of length $T$ with $D$ features to latent representation
   - Uses convolutional layers to capture temporal patterns

2. **Prototype Layer** containing $K$ learnable prototypes $\{p_1, p_2, ..., p_K\}$
   - Each prototype $p_k \in \mathbb{R}^H$ represents a characteristic market pattern
   - Prototypes are learned during training

3. **Classification Head** $g: \mathbb{R}^K \rightarrow \mathbb{R}^C$
   - Maps similarity scores to class probabilities
   - Typically a linear layer with softmax activation

### Similarity Functions

The similarity between an input representation $z = f(x)$ and prototype $p_k$ is computed using:

**Squared L2 Distance (converted to similarity)**:
$$s_k(x) = \log\left(\frac{||z - p_k||^2 + 1}{||z - p_k||^2 + \epsilon}\right)$$

**Cosine Similarity**:
$$s_k(x) = \frac{z \cdot p_k}{||z|| \cdot ||p_k||}$$

### Training Objective

The total loss function combines:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{clst} + \lambda_2 \mathcal{L}_{sep}$$

Where:
- $\mathcal{L}_{CE}$: Cross-entropy classification loss
- $\mathcal{L}_{clst}$: Clustering loss - encourages prototypes to be close to training examples
- $\mathcal{L}_{sep}$: Separation loss - encourages prototypes of different classes to be distant

## Market State Classification

### Defining Market States

We define five primary market states for classification:

| State | Description | Typical Characteristics |
|-------|-------------|------------------------|
| **Bullish Trend** | Strong upward momentum | Higher highs, higher lows, above MA |
| **Bearish Trend** | Strong downward momentum | Lower highs, lower lows, below MA |
| **Consolidation** | Range-bound market | Low volatility, price within bands |
| **Breakout** | Transition from consolidation | Volume spike, band breach |
| **Mean Reversion** | Return to equilibrium | Extreme RSI, price at band extremes |

### Feature Engineering for Prototypes

Input features for prototype learning include:

**Price-based Features**:
- Returns (1-period, 5-period, 20-period)
- Log price relative to moving averages
- Bollinger Band position

**Momentum Indicators**:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Rate of Change (ROC)

**Volatility Features**:
- ATR (Average True Range)
- Bollinger Band Width
- Historical Volatility

**Volume Features**:
- Volume ratio (current / average)
- On-Balance Volume trend
- Volume-Price Trend

## Implementation

### Code Example: Python Implementation

The Python implementation provides a complete prototype learning pipeline:

```
python/
├── __init__.py
├── model.py          # ProtoPNet model architecture
├── train.py          # Training pipeline
├── backtest.py       # Backtesting framework
├── data_loader.py    # Data loading utilities
└── notebooks/
    └── prototype_learning_example.ipynb
```

Key components:

1. **model.py**: Implements the ProtoPNet architecture with:
   - Convolutional feature encoder
   - Learnable prototype layer
   - Similarity computation
   - Classification head

2. **train.py**: Training pipeline with:
   - Combined loss function
   - Prototype projection (push to nearest training example)
   - Early stopping and model checkpointing

3. **backtest.py**: Backtesting with:
   - Signal generation from prototype similarities
   - Position sizing based on confidence
   - Performance metrics calculation

See [prototype_learning_example.ipynb](python/notebooks/prototype_learning_example.ipynb) for a complete walkthrough.

### Code Example: Rust Implementation

The Rust implementation provides high-performance inference for production:

```
rust/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs      # Bybit API client
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs  # Data preprocessing
│   │   └── features.rs   # Feature engineering
│   ├── models/
│   │   ├── mod.rs
│   │   └── prototype.rs  # Prototype network inference
│   └── metrics/
│       ├── mod.rs
│       ├── classification.rs
│       └── trading.rs
└── examples/
    ├── fetch_data.rs
    ├── prototype_classification.rs
    └── live_trading.rs
```

## Trading Strategy

### Signal Generation

The trading strategy uses prototype similarities to generate signals:

1. **State Classification**: Classify current market state based on prototype similarities
2. **Confidence Filtering**: Only trade when similarity to winning prototype exceeds threshold
3. **Position Sizing**: Scale position size by prediction confidence

```python
def generate_signal(similarities, threshold=0.7):
    max_similarity = similarities.max()
    predicted_state = similarities.argmax()

    if max_similarity < threshold:
        return 0  # No clear signal

    if predicted_state in [MarketState.BULLISH, MarketState.BREAKOUT_UP]:
        return 1  # Long
    elif predicted_state in [MarketState.BEARISH, MarketState.BREAKOUT_DOWN]:
        return -1  # Short
    else:
        return 0  # Neutral
```

### Backtesting Framework

The backtesting framework evaluates strategy performance:

- **Walk-forward optimization**: Train on rolling windows
- **Transaction costs**: Include slippage and commissions
- **Risk management**: Stop-loss and take-profit levels
- **Performance attribution**: Analyze which prototypes drive returns

## Data Sources

### Stock Market Data

Stock market data is fetched using `yfinance`:

```python
import yfinance as yf

# Fetch daily data for multiple symbols
symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
data = yf.download(symbols, start='2020-01-01', end='2024-01-01')
```

### Cryptocurrency Data (Bybit)

Cryptocurrency data is fetched from Bybit exchange API:

```python
from python.data_loader import BybitDataLoader

loader = BybitDataLoader()
btc_data = loader.get_klines(
    symbol='BTCUSDT',
    interval='1h',
    start_time='2023-01-01',
    end_time='2024-01-01'
)
```

The Rust implementation provides efficient data fetching:

```rust
use prototype_learning::api::BybitClient;

let client = BybitClient::new();
let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None)?;
```

## Evaluation Metrics

### Classification Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Detailed class-wise performance |
| **Prototype Purity** | How well prototypes represent their class |

### Trading Metrics

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted return |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Calmar Ratio** | CAGR / Maximum Drawdown |

### Interpretability Metrics

| Metric | Description |
|--------|-------------|
| **Prototype Diversity** | How different prototypes are from each other |
| **Activation Sparsity** | How often each prototype is "active" |
| **Nearest Example Distance** | How close prototypes are to real examples |

## References

1. **This Looks Like That: Deep Learning for Interpretable Image Recognition**
   - Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019)
   - URL: https://arxiv.org/abs/1806.10574
   - Introduces ProtoPNet architecture for interpretable classification

2. **Interpretable Machine Learning for Financial Risk Management**
   - Rudin, C. (2019)
   - URL: https://arxiv.org/abs/1811.10154
   - Discusses importance of interpretability in finance

3. **Machine Learning for Asset Managers**
   - López de Prado, M. (2020)
   - Cambridge University Press
   - Comprehensive guide to ML in finance

4. **Technical Analysis of the Financial Markets**
   - Murphy, J. J. (1999)
   - New York Institute of Finance
   - Foundation for technical indicators used as features
