# Chapter 238: Disentangled VAE Trading (Rust)

Rust implementation of disentangled Variational Autoencoders for learning
interpretable latent factors from market data.

## Disentanglement Methods

- **BetaVAE**: Weighted KL divergence with beta > 1
- **FactorVAE**: Total correlation penalty via discriminator
- **DIP-VAE**: Covariance penalty on inferred moments
- **BetaTCVAE**: Decomposed KL with total correlation term

## Quick Start

```bash
# Build
cargo build

# Run tests
cargo test

# Fetch market data
cargo run --example fetch_data

# Train model on synthetic data
cargo run --example train

# Run backtest
cargo run --example backtest
```

## Project Structure

- `src/api/` - Bybit API client
- `src/data/` - Data loading, feature engineering, dataset
- `src/model/` - Encoder, decoder, VAE with disentanglement losses
- `src/strategy/` - Signal generation and backtesting
- `examples/` - Usage examples

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| latent_dim | 8 | Latent space dimensionality |
| hidden_dim | 128 | Hidden layer size |
| beta | 4.0 | KL divergence weight |
| seq_len | 60 | Input sequence length |
| prediction_horizon | 12 | Lookahead periods |
