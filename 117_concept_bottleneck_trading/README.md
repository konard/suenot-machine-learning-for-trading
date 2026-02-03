# Chapter 117: Concept Bottleneck Trading

## Overview

Concept Bottleneck Models (CBMs) are interpretable machine learning architectures that make predictions through human-understandable intermediate concepts. In trading applications, CBMs provide transparency by first predicting interpretable market concepts (like trend strength, volatility regime, or momentum) before making final trading decisions.

This approach addresses a critical limitation of black-box models in finance: the inability to understand *why* a model makes specific trading decisions.

## Key Concepts

### What is a Concept Bottleneck Model?

A CBM consists of two stages:
1. **Concept Predictor**: Maps raw inputs X to interpretable concepts C
2. **Label Predictor**: Maps concepts C to final predictions Y

```
Raw Data (X) → [Concept Model] → Concepts (C) → [Task Model] → Predictions (Y)
```

### Benefits for Trading

- **Interpretability**: Understand which market conditions drive trading signals
- **Intervention**: Override incorrect concept predictions at inference time
- **Debugging**: Identify when and why the model fails
- **Regulatory Compliance**: Explainable decisions for audit requirements
- **Risk Management**: Human oversight of automated trading decisions

## Architecture

### Trading Concept Examples

| Concept | Description | Computation |
|---------|-------------|-------------|
| `trend_strength` | Current trend magnitude | Linear regression slope over window |
| `volatility_regime` | High/Low/Normal volatility | Rolling std vs historical percentiles |
| `momentum_signal` | Positive/Negative momentum | RSI, MACD crossover signals |
| `volume_profile` | Volume relative to average | Current volume / SMA(volume) |
| `support_resistance` | Near support/resistance levels | Distance to pivot points |
| `market_regime` | Bull/Bear/Sideways | Trend + volatility combination |

### Model Architecture

```python
class ConceptBottleneckTrader:
    def __init__(self):
        self.concept_encoder = ConceptEncoder(input_dim, concept_dim)
        self.task_predictor = TaskPredictor(concept_dim, output_dim)

    def forward(self, x):
        concepts = self.concept_encoder(x)  # Interpretable bottleneck
        prediction = self.task_predictor(concepts)
        return prediction, concepts
```

## Implementation

### Project Structure

```
117_concept_bottleneck_trading/
├── README.md                    # This file
├── README.ru.md                 # Russian documentation
├── readme.simple.md             # Simplified English
├── readme.simple.ru.md          # Simplified Russian
├── python/
│   ├── __init__.py
│   ├── concepts.py              # Concept computation module
│   ├── model.py                 # CBM model implementation
│   ├── data_loader.py           # Bybit data fetching
│   ├── backtest.py              # Backtesting framework
│   └── train.py                 # Training script
└── rust_examples/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── api/                 # Bybit API client
    │   ├── data/                # Data processing
    │   ├── models/              # CBM implementation
    │   └── utils/               # Technical indicators
    └── examples/
        ├── basic_cbm.rs
        └── trading_cbm.rs
```

## Data Sources

### Bybit API Integration

The implementation uses Bybit exchange for cryptocurrency data:

```python
# Python example
from data_loader import BybitDataLoader

loader = BybitDataLoader()
df = loader.fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    limit=1000
)
```

```rust
// Rust example
use concept_bottleneck_trading::api::BybitClient;

let client = BybitClient::new();
let klines = client.fetch_klines("BTCUSDT", "1h", 1000).await?;
```

### Supported Data

- **Cryptocurrency**: Bybit perpetual futures (BTCUSDT, ETHUSDT, etc.)
- **Stock Market**: Integration with Yahoo Finance for equities
- **Timeframes**: 1m, 5m, 15m, 1h, 4h, 1D

## Training Process

### 1. Concept Label Generation

Concepts can be generated through:
- **Rule-based**: Technical indicators with predefined thresholds
- **Expert annotation**: Manual labeling by traders
- **Clustering**: Unsupervised discovery of market regimes

### 2. Joint Training

```python
# Loss = Concept Loss + Task Loss
loss = concept_loss(pred_concepts, true_concepts) + \
       alpha * task_loss(pred_signal, true_signal)
```

### 3. Intervention at Inference

```python
# Override a concept if expert knowledge suggests it's wrong
concepts = model.predict_concepts(market_data)
if expert_override:
    concepts['volatility_regime'] = 'high'  # Manual correction
signal = model.predict_signal(concepts)
```

## Evaluation Metrics

### Concept Quality
- Concept prediction accuracy
- Concept-target correlation

### Trading Performance
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

## Research References

1. Koh, P. W., et al. (2020). "Concept Bottleneck Models." *ICML 2020*.
2. Chen, Z., et al. (2020). "Concept Whitening for Interpretable Image Recognition."
3. Losch, M., et al. (2019). "Interpretability Beyond Feature Attribution."

## Quick Start

### Python

```bash
cd 117_concept_bottleneck_trading/python
pip install -r requirements.txt
python train.py --symbol BTCUSDT --interval 1h
```

### Rust

```bash
cd 117_concept_bottleneck_trading/rust_examples
cargo run --example trading_cbm
```

## License

This project is part of the Machine Learning for Trading repository.
