# DCGAN for Cryptocurrency Time Series (Rust)

Modular implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for generating synthetic cryptocurrency price data using Bybit exchange data.

## Features

- **Data Fetching**: Async client for Bybit API v5 with pagination support
- **DCGAN Model**: PyTorch-based (via `tch-rs`) Generator and Discriminator
- **Training Pipeline**: Complete training loop with metrics, checkpointing, and label smoothing
- **Modular Design**: Separate modules for data, model, training, and utilities

## Project Structure

```
rust_dcgan_bybit/
├── Cargo.toml                 # Dependencies and project config
├── src/
│   ├── lib.rs                 # Library exports
│   ├── main.rs                # Main CLI application
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit_client.rs    # Bybit API client
│   │   ├── ohlcv.rs           # OHLCV data structures
│   │   ├── loader.rs          # DataLoader for batching
│   │   └── preprocessing.rs   # Normalization, sequences
│   ├── model/
│   │   ├── mod.rs
│   │   ├── generator.rs       # Generator network
│   │   ├── discriminator.rs   # Discriminator network
│   │   └── dcgan.rs           # Combined DCGAN model
│   ├── training/
│   │   ├── mod.rs
│   │   ├── trainer.rs         # Training loop
│   │   ├── losses.rs          # Loss functions
│   │   └── metrics.rs         # Training metrics
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── config.rs          # Configuration handling
│   │   └── checkpoint.rs      # Save/load checkpoints
│   └── bin/
│       ├── fetch_data.rs      # Data fetching binary
│       ├── train_model.rs     # Training binary
│       └── generate_samples.rs # Generation binary
└── README.md
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
cd rust_dcgan_bybit
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
    "sequence_length": 24,
    "batch_size": 64
  },
  "model": {
    "latent_dim": 100,
    "num_features": 5
  },
  "training": {
    "epochs": 100,
    "gen_lr": 0.0002,
    "disc_lr": 0.0002
  }
}
```

### 2. Fetch Data from Bybit

```bash
# Using main CLI
cargo run --release -- fetch --symbol BTCUSDT --interval 60 --days 30

# Using dedicated binary
cargo run --release --bin fetch_data -- \
  --symbol BTCUSDT \
  --interval 60 \
  --days 30 \
  --output data/btcusdt_1h_30d.csv
```

Available intervals: `1`, `5`, `15`, `30`, `60`, `120`, `240`, `D`, `W`

### 3. Train the Model

```bash
# Using main CLI
cargo run --release -- train --data data/btcusdt_1h_30d.csv --epochs 100

# Using dedicated binary with more options
cargo run --release --bin train_model -- \
  --data data/btcusdt_1h_30d.csv \
  --epochs 100 \
  --batch-size 64 \
  --sequence-length 24 \
  --latent-dim 100 \
  --gen-lr 0.0002 \
  --disc-lr 0.0002 \
  --checkpoint-dir checkpoints \
  --gpu  # if CUDA available
```

### 4. Generate Synthetic Samples

```bash
# Using main CLI
cargo run --release -- generate \
  --model checkpoints \
  --num-samples 100 \
  --output synthetic_samples.csv

# Using dedicated binary with more options
cargo run --release --bin generate_samples -- \
  --model checkpoints \
  --num-samples 100 \
  --sequence-length 24 \
  --denormalize \
  --output synthetic_btcusdt.csv

# Generate interpolated samples (smooth transitions)
cargo run --release --bin generate_samples -- \
  --model checkpoints \
  --num-samples 10 \
  --interpolate \
  --interp-steps 20 \
  --output interpolated_samples.csv
```

### 5. Evaluate Results

```bash
cargo run --release -- evaluate \
  --real data/btcusdt_1h_30d.csv \
  --synthetic synthetic_samples.csv
```

## API Usage

```rust
use rust_dcgan_bybit::{
    data::{BybitClient, DataLoader, OHLCVDataset, create_sequences, normalize_data},
    model::DCGAN,
    training::{Trainer, TrainingConfig},
};
use tch::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let dataset = client
        .fetch_historical_klines("BTCUSDT", "60", start_time, end_time)
        .await?;

    // Preprocess
    let features = dataset.to_feature_matrix();
    let data = /* convert to ndarray */;
    let (normalized, params) = normalize_data(&data);
    let sequences = create_sequences(&normalized, 24, 1);

    // Create model
    let device = Device::Cpu;
    let mut model = DCGAN::with_defaults(24, 5, 100, device);

    // Train
    let mut loader = DataLoader::new(sequences, 64, true, true);
    let config = TrainingConfig::default();
    let mut trainer = Trainer::new(config, device);
    trainer.train(&mut model, &mut loader);

    // Generate
    let synthetic = model.generate(100);

    Ok(())
}
```

## Model Architecture

### Generator
```
Input: (batch, 100)  # Latent noise vector
  → Dense(100, 768)
  → Reshape(256, 3)
  → ConvTranspose1D(256→128, k=4, s=2) + BN + LeakyReLU
  → ConvTranspose1D(128→64, k=4, s=2) + BN + LeakyReLU
  → ConvTranspose1D(64→32, k=4, s=2) + BN + LeakyReLU
  → ConvTranspose1D(32→5, k=3, s=1) + Tanh
Output: (batch, 24, 5)  # OHLCV sequence
```

### Discriminator
```
Input: (batch, 24, 5)  # OHLCV sequence
  → Conv1D(5→64, k=4, s=2) + LeakyReLU + Dropout(0.3)
  → Conv1D(64→128, k=4, s=2) + LeakyReLU + Dropout(0.3)
  → Conv1D(128→256, k=4, s=2) + LeakyReLU + Dropout(0.3)
  → Conv1D(256→512, k=4, s=2) + LeakyReLU + Dropout(0.3)
  → Flatten → Dense(1)
Output: (batch, 1)  # Real/Fake logit
```

## Training Tips

1. **Data Quantity**: Use at least 30 days of hourly data (~720 samples)
2. **Batch Size**: Start with 64, reduce if memory issues
3. **Learning Rate**: 2e-4 works well for both G and D
4. **Label Smoothing**: Enabled by default (real=0.9, fake=0.1)
5. **Mode Collapse**: If D_loss → 0 and G_loss → ∞, try:
   - Lower D learning rate
   - Increase D dropout
   - Add noise to D inputs

## Output Format

Generated samples are saved as CSV:

```csv
sample_id,timestep,open,high,low,close,volume
0,0,0.123456,0.234567,0.012345,0.123456,0.456789
0,1,0.234567,0.345678,0.123456,0.234567,0.567890
...
```

Values are normalized to [-1, 1]. Use `--denormalize` to get original scale.

## Dependencies

- `tch` - PyTorch bindings
- `ndarray` - N-dimensional arrays
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde` / `serde_json` - Serialization
- `clap` - CLI parsing
- `tracing` - Logging

## License

MIT

## References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [TimeGAN Paper](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf)
- [Bybit API v5](https://bybit-exchange.github.io/docs/v5/intro)
