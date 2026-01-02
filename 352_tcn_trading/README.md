# Chapter 352: Temporal Convolutional Networks (TCN) for Trading

## Overview

Temporal Convolutional Networks (TCN) represent a revolutionary approach to sequence modeling that has proven to be highly effective for financial time series prediction. Unlike traditional RNNs and LSTMs, TCNs use causal convolutions with dilations to capture long-range dependencies while maintaining parallelism and stable gradients.

## Trading Strategy

**Core Strategy:** Using TCN to predict cryptocurrency price movements and generate trading signals based on multi-scale temporal patterns.

**Key Advantages:**
1. **Parallelism** — Unlike RNNs, all time steps can be computed in parallel during training
2. **Stable Gradients** — No vanishing/exploding gradient problems common in RNNs
3. **Flexible Receptive Field** — Dilation allows exponential increase in receptive field
4. **Causal Architecture** — No information leakage from future to past

**Edge:** TCNs can capture both short-term microstructure patterns and long-term trends simultaneously through dilated convolutions.

## Technical Specification

### Architecture Components

| Component | Description |
|-----------|-------------|
| Causal Convolution | Ensures output at time t depends only on inputs at times ≤ t |
| Dilated Convolution | Exponentially increases receptive field without increasing parameters |
| Residual Connections | Enables training of deep networks |
| Weight Normalization | Stabilizes training |

### TCN Block Structure

```
Input → Conv1D(dilation=1) → ReLU → Dropout →
        Conv1D(dilation=1) → ReLU → Dropout →
        + Residual Connection → Output
```

Multiple blocks are stacked with increasing dilation rates: 1, 2, 4, 8, 16, ...

### Mathematical Foundation

#### Causal Convolution

For input sequence **x** and filter **f** of size k:
```
(x *_c f)(t) = Σ_{i=0}^{k-1} f(i) · x(t - i)
```

Key property: Output at time t depends only on x(t), x(t-1), ..., x(t-k+1)

#### Dilated Convolution

For dilation factor d:
```
(x *_d f)(t) = Σ_{i=0}^{k-1} f(i) · x(t - d·i)
```

With dilation, receptive field grows exponentially: **R = 1 + (k-1) · Σ d_i**

#### Receptive Field Calculation

For L layers with dilation doubling:
```
Receptive Field = 1 + 2·(k-1)·(2^L - 1)
```

Example: k=3, L=8 → RF = 1 + 2·2·255 = 1021 time steps

### Key Metrics

| Metric | Description |
|--------|-------------|
| Directional Accuracy | Correct prediction of price movement direction |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss |

## Content

1. [TCN Architecture](#tcn-architecture)
2. [Why TCN for Trading](#why-tcn-for-trading)
3. [Implementation Details](#implementation-details)
4. [Feature Engineering](#feature-engineering)
5. [Trading Signal Generation](#trading-signal-generation)
6. [Backtesting Framework](#backtesting-framework)
7. [Rust Implementation](#rust-implementation)
8. [References](#references)

## TCN Architecture

### Core Components

#### 1. Causal Convolution Layer

Causal convolutions ensure that predictions at time t only use information available at time t or earlier:

```
Time:     t-4   t-3   t-2   t-1    t
           │     │     │     │     │
           └──┬──┴──┬──┴──┬──┴──┬──┘
              │     │     │     │
           [Conv1D, kernel_size=2, dilation=1]
              │     │     │     │
              └──┬──┴──┬──┴──┬──┘
                 │     │     │
              [Conv1D, kernel_size=2, dilation=2]
                 │     │     │
                 └──┬──┴──┬──┘
                    │     │
              [Conv1D, kernel_size=2, dilation=4]
                    │
                    ▼
              Prediction for time t
```

#### 2. Residual Block

```rust
struct TCNResidualBlock {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
    relu: ReLU,
    dropout: Dropout,
    residual_conv: Option<Conv1d>,  // if input/output channels differ
}
```

#### 3. Full TCN Network

```rust
struct TCN {
    input_projection: Linear,
    residual_blocks: Vec<TCNResidualBlock>,
    output_projection: Linear,
}
```

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| num_channels | [64, 64, 64, 64] | Channels per layer |
| kernel_size | 3 | Convolution kernel size |
| dropout | 0.2 | Dropout probability |
| dilation_base | 2 | Dilation factor multiplier |

## Why TCN for Trading

### Advantages Over RNNs/LSTMs

| Aspect | RNN/LSTM | TCN |
|--------|----------|-----|
| Parallelism | Sequential processing | Fully parallel |
| Gradient Flow | Vanishing/exploding | Stable (residual) |
| Memory | Bounded by hidden state | Explicit via receptive field |
| Training Speed | Slow (sequential) | Fast (parallel) |
| Long Dependencies | Difficult | Natural (dilations) |

### Trading-Specific Benefits

1. **Multi-Scale Pattern Recognition**
   - Small dilations capture tick-level patterns (microstructure)
   - Large dilations capture trend patterns (macro movements)

2. **Real-Time Inference**
   - Single forward pass for prediction
   - Constant inference time regardless of sequence length

3. **Interpretability**
   - Receptive field is explicit and controllable
   - Attention visualization possible via gradient analysis

## Implementation Details

### Input Features

```rust
struct TradingFeatures {
    // Price-based
    returns: Vec<f64>,           // Log returns
    volatility: Vec<f64>,        // Rolling volatility

    // Technical indicators
    rsi: Vec<f64>,               // Relative Strength Index
    macd: Vec<f64>,              // MACD line
    macd_signal: Vec<f64>,       // MACD signal line
    bollinger_upper: Vec<f64>,   // Bollinger upper band
    bollinger_lower: Vec<f64>,   // Bollinger lower band

    // Volume features
    volume_ratio: Vec<f64>,      // Volume / Average volume
    obv: Vec<f64>,               // On-Balance Volume

    // Order flow (if available)
    bid_ask_imbalance: Vec<f64>, // Bid-ask volume imbalance
    trade_flow: Vec<f64>,        // Net buy/sell flow
}
```

### Target Variable Options

1. **Classification (Direction)**
   - Classes: Up, Down, Neutral
   - Threshold-based: |return| > threshold → directional

2. **Regression (Return Magnitude)**
   - Predict actual return value
   - Useful for position sizing

3. **Multi-Horizon**
   - Predict returns at multiple future horizons
   - [1 bar, 5 bars, 15 bars, 60 bars]

### Loss Functions

```rust
enum TradingLoss {
    // Classification
    CrossEntropy,
    FocalLoss { gamma: f64 },  // For imbalanced classes

    // Regression
    MSE,
    Huber { delta: f64 },      // Robust to outliers
    Quantile { tau: f64 },     // For prediction intervals

    // Trading-Specific
    SharpeRatioLoss,           // Maximize Sharpe directly
    DirectionalAccuracy,       // Sign(predicted) == Sign(actual)
}
```

## Feature Engineering

### Technical Indicators

```rust
impl TechnicalIndicators {
    pub fn calculate_all(&self, candles: &[Candle]) -> FeatureMatrix {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        FeatureMatrix {
            // Momentum
            rsi_14: self.rsi(&closes, 14),
            rsi_7: self.rsi(&closes, 7),
            macd: self.macd(&closes, 12, 26, 9),
            momentum_10: self.momentum(&closes, 10),
            roc_10: self.rate_of_change(&closes, 10),

            // Volatility
            atr_14: self.atr(&highs, &lows, &closes, 14),
            bollinger: self.bollinger_bands(&closes, 20, 2.0),
            keltner: self.keltner_channels(&highs, &lows, &closes, 20),

            // Trend
            sma_20: self.sma(&closes, 20),
            sma_50: self.sma(&closes, 50),
            ema_12: self.ema(&closes, 12),
            ema_26: self.ema(&closes, 26),
            adx_14: self.adx(&highs, &lows, &closes, 14),

            // Volume
            obv: self.on_balance_volume(&closes, &volumes),
            vwap: self.vwap(&highs, &lows, &closes, &volumes),
            volume_sma: self.sma(&volumes, 20),

            // Price action
            candle_patterns: self.detect_patterns(&candles),
            support_resistance: self.find_sr_levels(&closes, 20),
        }
    }
}
```

### Normalization Strategy

```rust
enum NormalizationMethod {
    // Z-score normalization (rolling window)
    ZScore { window: usize },

    // Min-max to [0, 1] (rolling window)
    MinMax { window: usize },

    // Robust scaling (median, IQR)
    Robust { window: usize },

    // Log transformation for prices
    LogReturns,

    // Percentage rank (0-100)
    PercentileRank { window: usize },
}
```

## Trading Signal Generation

### Signal Generation Pipeline

```rust
struct SignalGenerator {
    tcn_model: TCN,
    threshold_long: f64,   // e.g., 0.6 for 60% confidence
    threshold_short: f64,  // e.g., 0.6 for 60% confidence
    position_sizer: PositionSizer,
}

impl SignalGenerator {
    pub fn generate_signal(&self, features: &FeatureMatrix) -> TradingSignal {
        // Get model prediction
        let prediction = self.tcn_model.predict(features);

        // Convert to probabilities (softmax for classification)
        let probs = softmax(&prediction);

        // Generate signal based on thresholds
        let signal = if probs.up > self.threshold_long {
            SignalType::Long
        } else if probs.down > self.threshold_short {
            SignalType::Short
        } else {
            SignalType::Neutral
        };

        // Calculate position size based on confidence
        let confidence = probs.max();
        let position_size = self.position_sizer.calculate(confidence);

        TradingSignal {
            signal_type: signal,
            confidence,
            position_size,
            predicted_return: prediction.expected_return,
            timestamp: Utc::now(),
        }
    }
}
```

### Risk Management Integration

```rust
struct RiskManager {
    max_position_size: f64,     // Maximum position as fraction of capital
    max_daily_loss: f64,        // Stop trading if daily loss exceeds this
    stop_loss_pct: f64,         // Per-trade stop loss
    take_profit_pct: f64,       // Per-trade take profit
    max_drawdown_pct: f64,      // Maximum drawdown before reducing exposure
}

impl RiskManager {
    pub fn validate_signal(&self, signal: &TradingSignal, state: &PortfolioState) -> ValidatedSignal {
        let mut adjusted = signal.clone();

        // Check daily loss limit
        if state.daily_pnl < -self.max_daily_loss * state.capital {
            return ValidatedSignal::Blocked("Daily loss limit reached");
        }

        // Reduce exposure if in drawdown
        if state.current_drawdown > self.max_drawdown_pct * 0.5 {
            adjusted.position_size *= 0.5;
        }

        // Apply maximum position limit
        adjusted.position_size = adjusted.position_size.min(self.max_position_size);

        // Set stop loss and take profit
        adjusted.stop_loss = Some(self.stop_loss_pct);
        adjusted.take_profit = Some(self.take_profit_pct);

        ValidatedSignal::Approved(adjusted)
    }
}
```

## Backtesting Framework

### Backtest Engine

```rust
struct BacktestEngine {
    initial_capital: f64,
    commission: f64,          // Per-trade commission
    slippage: f64,            // Estimated slippage
    margin_requirement: f64,  // For leveraged trading
}

impl BacktestEngine {
    pub fn run(&self, signals: &[TradingSignal], prices: &[Candle]) -> BacktestResult {
        let mut capital = self.initial_capital;
        let mut position = 0.0;
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();

        for (i, (signal, candle)) in signals.iter().zip(prices).enumerate() {
            // Execute signal
            if signal.signal_type != SignalType::Neutral {
                let trade = self.execute_trade(signal, candle, &mut capital, &mut position);
                trades.push(trade);
            }

            // Mark-to-market
            let equity = capital + position * candle.close;
            equity_curve.push(EquityPoint {
                timestamp: candle.timestamp,
                equity,
                position,
            });
        }

        // Calculate metrics
        BacktestResult {
            total_return: (capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio: self.calculate_sharpe(&equity_curve),
            sortino_ratio: self.calculate_sortino(&equity_curve),
            max_drawdown: self.calculate_max_drawdown(&equity_curve),
            win_rate: self.calculate_win_rate(&trades),
            profit_factor: self.calculate_profit_factor(&trades),
            total_trades: trades.len(),
            avg_trade_duration: self.calculate_avg_duration(&trades),
            equity_curve,
            trades,
        }
    }
}
```

### Walk-Forward Validation

```rust
struct WalkForwardValidator {
    train_period: usize,    // e.g., 1000 bars
    test_period: usize,     // e.g., 200 bars
    retrain_frequency: usize, // Retrain every N bars
}

impl WalkForwardValidator {
    pub fn validate(&self, data: &MarketData, model_factory: &TCNFactory) -> ValidationResult {
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();

        let mut start = 0;
        while start + self.train_period + self.test_period <= data.len() {
            // Split data
            let train_data = &data[start..start + self.train_period];
            let test_data = &data[start + self.train_period..start + self.train_period + self.test_period];

            // Train model
            let model = model_factory.create_and_train(train_data);

            // Generate predictions on test set
            let predictions = model.predict_batch(test_data);
            let actuals = self.extract_targets(test_data);

            all_predictions.extend(predictions);
            all_actuals.extend(actuals);

            // Move window
            start += self.retrain_frequency;
        }

        ValidationResult {
            predictions: all_predictions,
            actuals: all_actuals,
            metrics: self.calculate_metrics(&all_predictions, &all_actuals),
        }
    }
}
```

## Rust Implementation

The complete Rust implementation is available in the `rust_tcn_trading/` directory:

```
rust_tcn_trading/
├── Cargo.toml
└── src/
    ├── lib.rs              # Library entry point
    ├── tcn/                 # TCN implementation
    │   ├── mod.rs
    │   ├── layer.rs         # Causal convolution layers
    │   ├── block.rs         # Residual blocks
    │   └── model.rs         # Full TCN model
    ├── features/            # Feature engineering
    │   ├── mod.rs
    │   ├── technical.rs     # Technical indicators
    │   └── normalize.rs     # Normalization
    ├── trading/             # Trading logic
    │   ├── mod.rs
    │   ├── signal.rs        # Signal generation
    │   ├── risk.rs          # Risk management
    │   └── backtest.rs      # Backtesting
    ├── api/                 # Bybit API client
    │   ├── mod.rs
    │   ├── client.rs
    │   └── types.rs
    ├── utils/               # Utilities
    │   ├── mod.rs
    │   └── metrics.rs       # Performance metrics
    └── bin/                 # Example binaries
        ├── fetch_data.rs    # Fetch crypto data from Bybit
        ├── train_tcn.rs     # Train TCN model
        ├── backtest.rs      # Run backtest
        └── live_signals.rs  # Generate live signals
```

### Quick Start

```bash
# Clone and navigate to the chapter
cd machine-learning-for-trading/352_tcn_trading/rust_tcn_trading

# Build the project
cargo build --release

# Fetch market data
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --limit 1000

# Train TCN model
cargo run --bin train_tcn -- --data data/btcusdt_1h.csv --epochs 100

# Run backtest
cargo run --bin backtest -- --model models/tcn.bin --data data/btcusdt_1h.csv
```

## References

### Original TCN Paper

1. **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling**
   - Authors: Shaojie Bai, J. Zico Kolter, Vladlen Koltun
   - URL: https://arxiv.org/abs/1803.01271
   - Year: 2018
   - Key finding: TCN outperforms canonical recurrent networks on diverse sequence modeling tasks

### Financial Applications

2. **Temporal Convolutional Networks for Financial Time Series**
   - Demonstrates TCN effectiveness for stock price prediction
   - Comparison with LSTM and Transformer models

3. **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books**
   - Uses CNN architecture for order book data
   - Similar principles applied to high-frequency trading

4. **Stock Price Prediction Using Temporal Convolution Network**
   - Application to various stock markets
   - Feature engineering best practices

### Related Architectures

5. **WaveNet: A Generative Model for Raw Audio**
   - DeepMind, 2016
   - Original dilated causal convolution architecture
   - https://arxiv.org/abs/1609.03499

6. **Attention Is All You Need (Transformer)**
   - Vaswani et al., 2017
   - Alternative approach to sequence modeling
   - https://arxiv.org/abs/1706.03762

### Trading and Market Microstructure

7. **Advances in Financial Machine Learning**
   - Marcos López de Prado, 2018
   - Comprehensive guide to ML in finance

8. **Machine Learning for Algorithmic Trading**
   - Stefan Jansen, 2020
   - Practical implementation guide

## Further Reading

- [TCN GitHub Repository](https://github.com/locuslab/TCN) - Original implementation
- [Keras TCN](https://github.com/philipperemy/keras-tcn) - Popular Keras implementation
- [PyTorch TCN](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py) - PyTorch reference

## License

MIT License - See LICENSE file for details.
