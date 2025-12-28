# Linear Models for Cryptocurrency Trading (Rust)

Rust implementation of linear models for cryptocurrency price prediction using data from **Bybit** exchange.

## Features

- 📊 **Bybit API Client** - Fetch OHLCV data for any cryptocurrency pair
- 🔧 **Feature Engineering** - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- 📈 **Linear Regression** - OLS and Gradient Descent implementations
- 🎯 **Regularization** - Ridge (L2), Lasso (L1), and Elastic Net
- 🔀 **Logistic Regression** - Binary classification for price direction
- 📉 **Metrics** - Comprehensive regression and classification metrics

## Project Structure

```
rust_examples/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Library entry point
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs        # Bybit API client
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs    # Data preprocessing utilities
│   │   └── features.rs     # Technical indicator calculations
│   ├── models/
│   │   ├── mod.rs
│   │   ├── linear.rs       # Linear regression (OLS, GD, SGD)
│   │   ├── regularization.rs  # Ridge, Lasso, Elastic Net
│   │   └── logistic.rs     # Logistic regression
│   └── metrics/
│       ├── mod.rs
│       ├── regression.rs   # MSE, RMSE, R², IC, etc.
│       └── classification.rs  # Accuracy, F1, AUC-ROC, etc.
└── examples/
    ├── fetch_data.rs       # Data fetching example
    ├── linear_regression.rs    # Linear regression example
    ├── ridge_lasso.rs      # Regularization example
    ├── logistic_regression.rs  # Classification example
    └── full_pipeline.rs    # Complete ML pipeline
```

## Installation

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- OpenBLAS (for linear algebra operations)

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# macOS
brew install openblas

# Fedora
sudo dnf install openblas-devel
```

### Build

```bash
cd rust_examples
cargo build --release
```

## Usage

### Run Examples

```bash
# Fetch cryptocurrency data from Bybit
cargo run --example fetch_data

# Linear regression for return prediction
cargo run --example linear_regression

# Ridge and Lasso regularization
cargo run --example ridge_lasso

# Logistic regression for direction prediction
cargo run --example logistic_regression

# Complete ML trading pipeline
cargo run --example full_pipeline
```

### Use as Library

```rust
use linear_models_crypto::{
    api::bybit::{BybitClient, Interval},
    data::features::FeatureEngineering,
    models::linear::LinearRegression,
    metrics::regression::RegressionMetrics,
};

fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(500), None, None)?;

    // Generate features
    let (features, feature_names) = FeatureEngineering::generate_features(&klines);
    let target = FeatureEngineering::create_target(&klines, 1);

    // Train model
    let mut model = LinearRegression::new(true);
    model.fit(&features, &target)?;

    // Evaluate
    let predictions = model.predict(&features)?;
    let metrics = RegressionMetrics::calculate(&target, &predictions);

    println!("R²: {:.4}", metrics.r2);
    println!("IC: {:.4}", metrics.ic);

    Ok(())
}
```

## Modules

### API (Bybit Client)

```rust
use linear_models_crypto::api::bybit::{BybitClient, Interval};

let client = BybitClient::new();

// Fetch 100 hourly candles
let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;

// Fetch historical data with pagination
let history = client.get_klines_history("ETHUSDT", Interval::Day1, start_time, end_time)?;

// Get current ticker
let ticker = client.get_ticker("SOLUSDT")?;
```

### Feature Engineering

Supported technical indicators:
- **Moving Averages**: SMA, EMA
- **Momentum**: RSI, MACD, Momentum, ROC
- **Volatility**: Bollinger Bands, ATR, Rolling Volatility
- **Volume**: OBV, Volume Ratio
- **Price Features**: Lagged returns, High-Low range

```rust
use linear_models_crypto::data::features::FeatureEngineering;

// Generate all features
let (features, names) = FeatureEngineering::generate_features(&klines);

// Individual indicators
let sma_20 = FeatureEngineering::sma(&prices, 20);
let rsi_14 = FeatureEngineering::rsi(&prices, 14);
let (macd, signal, hist) = FeatureEngineering::macd(&prices, 12, 26, 9);
```

### Models

#### Linear Regression

```rust
use linear_models_crypto::models::linear::{LinearRegression, LinearRegressionGD};

// OLS
let mut ols = LinearRegression::new(true);
ols.fit(&x_train, &y_train)?;
let predictions = ols.predict(&x_test)?;

// Gradient Descent
let mut gd = LinearRegressionGD::new(0.01, 1000, 1e-6, true);
gd.fit(&x_train, &y_train)?;
```

#### Regularized Regression

```rust
use linear_models_crypto::models::regularization::{RidgeRegression, LassoRegression, ElasticNet};

// Ridge (L2)
let mut ridge = RidgeRegression::new(1.0, true, false);
ridge.fit(&x_train, &y_train)?;

// Lasso (L1) - for feature selection
let mut lasso = LassoRegression::new(0.01, true, 1000, 1e-6);
lasso.fit(&x_train, &y_train)?;
println!("Selected features: {:?}", lasso.selected_features());

// Elastic Net
let mut enet = ElasticNet::new(0.1, 0.5, true, 1000, 1e-6);
enet.fit(&x_train, &y_train)?;
```

#### Logistic Regression

```rust
use linear_models_crypto::models::logistic::{LogisticRegression, Regularization};

// Basic
let mut lr = LogisticRegression::default();
lr.fit(&x_train, &y_train)?;

// With L2 regularization
let mut lr_l2 = LogisticRegression::with_l2(1.0);
lr_l2.fit(&x_train, &y_train)?;

// Predictions
let probabilities = lr.predict_proba(&x_test)?;
let classes = lr.predict(&x_test)?;
```

### Metrics

```rust
use linear_models_crypto::metrics::{
    regression::RegressionMetrics,
    classification::ClassificationMetrics,
};

// Regression
let reg_metrics = RegressionMetrics::calculate(&y_true, &y_pred);
println!("{}", reg_metrics.report());

// Classification
let clf_metrics = ClassificationMetrics::calculate_with_proba(&y_true, &y_pred, Some(&y_proba));
println!("{}", clf_metrics.report());
```

## Key Metrics

### Regression
- **MSE/RMSE**: Mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **IC**: Information coefficient (Pearson correlation)
- **Hit Rate**: Directional accuracy

### Classification
- **Accuracy, Precision, Recall, F1**
- **AUC-ROC**: Area under ROC curve
- **MCC**: Matthews correlation coefficient
- **Log Loss**: Binary cross-entropy

## Performance Tips

1. **Use release mode** for 10-100x speedup:
   ```bash
   cargo run --release --example full_pipeline
   ```

2. **Standardize features** before fitting regularized models

3. **Use time series cross-validation** to avoid lookahead bias

4. **Start with simpler models** (OLS) before adding regularization

## Contributing

Contributions are welcome! Please ensure:
- Code passes `cargo clippy`
- Tests pass with `cargo test`
- Format with `cargo fmt`

## License

MIT License - See main repository for details.
