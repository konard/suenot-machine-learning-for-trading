//! # Disentangled VAE Trading
//!
//! Learn interpretable latent factors from market data using disentangled
//! Variational Autoencoders. Supports multiple disentanglement methods
//! including beta-VAE, FactorVAE, DIP-VAE, and beta-TCVAE.
//!
//! ## Features
//!
//! - Disentangled latent representations of market regimes
//! - Multiple disentanglement methods (BetaVAE, FactorVAE, DIPVAE, BetaTCVAE)
//! - Reparameterization trick for end-to-end training
//! - Bybit API integration for cryptocurrency data
//! - Signal generation from latent factor analysis
//!
//! ## Modules
//!
//! - `api` - Client for Bybit API
//! - `data` - Data loading and preprocessing
//! - `model` - Disentangled VAE architecture
//! - `strategy` - Trading strategy and backtesting
//!
//! ## Example
//!
//! ```no_run
//! use disentangled_vae::{BybitClient, DataLoader, DisentangledVAE, DisentangledVAEConfig, DisentanglementMethod};
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 1000).await.unwrap();
//!
//!     let loader = DataLoader::new()
//!         .seq_len(60)
//!         .target_horizon(12);
//!     let dataset = loader.prepare_dataset(&klines).unwrap();
//!
//!     let config = DisentangledVAEConfig {
//!         latent_dim: 8,
//!         hidden_dim: 128,
//!         beta: 4.0,
//!         method: DisentanglementMethod::BetaVAE,
//!         ..Default::default()
//!     };
//!     let mut vae = DisentangledVAE::new(config);
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, BybitError, Kline, OrderBook, Ticker};
pub use data::{DataLoader, Dataset, Features};
pub use model::{
    Decoder, DisentangledVAE, DisentangledVAEConfig, DisentanglementMethod, Encoder,
};
pub use strategy::{BacktestConfig, BacktestResult, Backtester, Signal, SignalGenerator, SignalGeneratorConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default settings
pub mod defaults {
    /// Latent space dimensionality
    pub const LATENT_DIM: usize = 8;

    /// Hidden layer dimension
    pub const HIDDEN_DIM: usize = 128;

    /// Beta weight for KL divergence
    pub const BETA: f64 = 4.0;

    /// Sequence length for input windows
    pub const SEQ_LEN: usize = 60;

    /// Prediction horizon
    pub const PREDICTION_HORIZON: usize = 12;

    /// Learning rate
    pub const LEARNING_RATE: f64 = 0.001;

    /// Batch size
    pub const BATCH_SIZE: usize = 32;

    /// Number of epochs
    pub const EPOCHS: usize = 100;
}
