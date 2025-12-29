# Diffusion Models for Cryptocurrency Forecasting (Rust)

Rust implementation of Denoising Diffusion Probabilistic Models (DDPM) for cryptocurrency price forecasting using Bybit exchange data.

## Features

- **Data Fetching**: Async client for Bybit API v5 with pagination support
- **DDPM Model**: PyTorch-based (via `tch-rs`) diffusion model
- **Noise Schedules**: Linear, cosine, and sigmoid schedules
- **Training Pipeline**: Complete training loop with metrics and checkpointing
- **Probabilistic Forecasting**: Monte Carlo sampling for uncertainty quantification
- **Modular Design**: Separate modules for data, model, training, and utilities

## Project Structure

```
rust_diffusion_crypto/
├── Cargo.toml                 # Dependencies and project config
├── README.md                  # This file
├── src/
│   ├── lib.rs                 # Library exports
│   ├── main.rs                # Main CLI application
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit_client.rs    # Bybit API client
│   │   ├── ohlcv.rs           # OHLCV data structures
│   │   ├── features.rs        # Technical indicators
│   │   └── preprocessing.rs   # Normalization, sequences
│   ├── model/
│   │   ├── mod.rs
│   │   ├── schedule.rs        # Noise schedules
│   │   ├── unet.rs            # U-Net architecture
│   │   └── ddpm.rs            # DDPM model
│   ├── training/
│   │   ├── mod.rs
│   │   ├── trainer.rs         # Training loop
│   │   ├── losses.rs          # Loss functions
│   │   └── metrics.rs         # Training metrics
│   └── utils/
│       ├── mod.rs
│       ├── config.rs          # Configuration handling
│       └── checkpoint.rs      # Save/load checkpoints
└── examples/
    ├── fetch_data.rs          # Data fetching example
    ├── train_ddpm.rs          # Training example
    └── forecast.rs            # Forecasting example
```

## Prerequisites

### LibTorch

This project requires LibTorch (PyTorch C++ API). Install it:

```bash
# Option 1: Download from PyTorch website
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-*.zip

# Set environment variable
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Option 2: Using conda
conda install pytorch cpuonly -c pytorch
```

### CUDA (Optional)

For GPU support:
```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

## Installation

```bash
cd rust_diffusion_crypto
cargo build --release
```

## Usage

### 1. Initialize Configuration

```bash
cargo run --release -- init --output config.json
```

This creates a default configuration file:

```json
{
  "data": {
    "symbol": "BTCUSDT",
    "interval": "60",
    "sequence_length": 100,
    "forecast_horizon": 24,
    "batch_size": 32
  },
  "model": {
    "hidden_dim": 256,
    "time_emb_dim": 64,
    "num_diffusion_steps": 1000,
    "noise_schedule": "cosine"
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.0001,
    "grad_clip": 1.0
  }
}
```

### 2. Fetch Data from Bybit

```bash
# Using main CLI
cargo run --release -- fetch --symbol BTCUSDT --interval 60 --days 90

# Using dedicated example
cargo run --release --example fetch_data -- \
  --symbol BTCUSDT \
  --interval 60 \
  --days 90 \
  --output data/btcusdt_1h_90d.csv
```

Available intervals: `1`, `5`, `15`, `30`, `60`, `120`, `240`, `D`, `W`

### 3. Train the Model

```bash
# Using main CLI
cargo run --release -- train --data data/btcusdt_1h_90d.csv --epochs 100

# Using dedicated example with more options
cargo run --release --example train_ddpm -- \
  --data data/btcusdt_1h_90d.csv \
  --epochs 100 \
  --batch-size 32 \
  --sequence-length 100 \
  --forecast-horizon 24 \
  --diffusion-steps 1000 \
  --learning-rate 0.0001 \
  --checkpoint-dir checkpoints \
  --gpu  # if CUDA available
```

Training output:
```
Epoch   1/100 | Loss: 0.982341 | LR: 0.0001
Epoch  20/100 | Loss: 0.234521 | LR: 0.0001
Epoch  40/100 | Loss: 0.089234 | LR: 0.0001
...
Training complete! Best loss: 0.045123
Model saved to: checkpoints/ddpm_final.pt
```

### 4. Generate Forecasts

```bash
# Using main CLI
cargo run --release -- forecast \
  --model checkpoints/ddpm_final.pt \
  --data data/btcusdt_1h_90d.csv \
  --num-samples 100 \
  --output forecasts.csv

# Using dedicated example
cargo run --release --example forecast -- \
  --model checkpoints/ddpm_final.pt \
  --context-length 100 \
  --forecast-horizon 24 \
  --num-samples 100 \
  --output forecasts.csv
```

Output includes probabilistic forecasts:
```csv
hour,mean,std,p5,p25,p50,p75,p95
1,45234.56,123.45,45012.34,45156.78,45234.56,45312.34,45456.78
2,45267.89,156.78,44989.01,45123.45,45267.89,45412.34,45546.67
...
```

## API Usage

```rust
use diffusion_crypto::{
    data::{BybitClient, OHLCVDataset, FeatureEngineer},
    model::{DDPM, NoiseSchedule},
    training::{Trainer, TrainingConfig},
};
use tch::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let ohlcv = client
        .fetch_historical_klines("BTCUSDT", "60", 90)
        .await?;

    // Compute features
    let engineer = FeatureEngineer::new();
    let features = engineer.compute_all(&ohlcv);

    // Create noise schedule
    let schedule = NoiseSchedule::cosine(1000);

    // Create model
    let device = Device::cuda_if_available();
    let model = DDPM::new(
        features.num_features(),
        100,   // sequence_length
        24,    // forecast_horizon
        256,   // hidden_dim
        &schedule,
        device
    );

    // Train
    let config = TrainingConfig::default();
    let mut trainer = Trainer::new(model, config, device);
    trainer.train(&features, 100)?;

    // Forecast
    let forecast = trainer.model().forecast(
        &features.last_sequence(),
        100  // num_samples for Monte Carlo
    )?;

    println!("Forecast mean: {:?}", forecast.mean);
    println!("Forecast std: {:?}", forecast.std);

    Ok(())
}
```

## Model Architecture

### Conditional DDPM

```
Input: x_noisy [batch, forecast_horizon], condition [batch, seq_len, features]
│
├── Time Embedding (Sinusoidal → MLP)
│   └── [batch, hidden_dim]
│
├── Condition Encoder (LSTM)
│   └── [batch, hidden_dim]
│
├── Combined: [x_noisy, time_emb, cond_emb]
│   └── [batch, forecast_horizon + 2*hidden_dim]
│
├── Denoising Network (MLP with residual connections)
│   ├── Linear → LayerNorm → SiLU → Dropout
│   ├── Linear → LayerNorm → SiLU → Dropout
│   ├── Linear → LayerNorm → SiLU → Dropout
│   └── Linear → LayerNorm → SiLU → Dropout
│
└── Output: noise_pred [batch, forecast_horizon]
```

### Noise Schedules

| Schedule | Description | Best For |
|----------|-------------|----------|
| Linear | β linearly from 0.0001 to 0.02 | General purpose |
| Cosine | Based on cosine annealing | Smaller sequences |
| Sigmoid | Smooth S-curve | Stable training |

## Training Tips

1. **Data Quantity**: Use at least 90 days of hourly data (~2160 samples)
2. **Sequence Length**: 100 hours captures weekly patterns
3. **Forecast Horizon**: 24 hours is a good balance
4. **Diffusion Steps**: 1000 for quality, 500 for speed
5. **Learning Rate**: Start with 1e-4, reduce if unstable
6. **Batch Size**: 32 works well, reduce if OOM

## Evaluation Metrics

The forecast output includes:
- **Mean**: Expected forecast value
- **Std**: Standard deviation (uncertainty)
- **Percentiles**: p5, p25, p50, p75, p95

Evaluation metrics:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **CRPS**: Continuous Ranked Probability Score (for probabilistic forecasts)

## Comparison with Python

| Aspect | Python | Rust |
|--------|--------|------|
| Training Speed | 1x | 1.5-2x faster |
| Inference | 1x | 2-3x faster |
| Memory | High | Lower |
| Deployment | Complex | Single binary |
| Development | Faster | More setup |

## Dependencies

- `tch` - PyTorch bindings for Rust
- `ndarray` - N-dimensional arrays
- `tokio` - Async runtime
- `reqwest` - HTTP client for Bybit API
- `serde` / `serde_json` - Serialization
- `clap` - CLI argument parsing
- `tracing` - Logging

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [TimeGrad: Autoregressive Denoising Diffusion](https://arxiv.org/abs/2101.12072)
- [CSDI: Conditional Score-based Diffusion](https://arxiv.org/abs/2107.03502)
- [Bybit API v5 Documentation](https://bybit-exchange.github.io/docs/v5/intro)
- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)

## License

MIT
