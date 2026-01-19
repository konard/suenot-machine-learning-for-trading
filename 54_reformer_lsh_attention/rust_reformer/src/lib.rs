//! # Reformer
//!
//! Rust implementation of the Reformer architecture with LSH Attention
//! for efficient long-sequence cryptocurrency prediction using Bybit data.
//!
//! ## Features
//!
//! - LSH (Locality-Sensitive Hashing) Attention with O(L·log(L)) complexity
//! - Multi-round hashing for improved accuracy
//! - Reversible layers for memory efficiency
//! - Bybit API integration for cryptocurrency data
//!
//! ## Modules
//!
//! - `api` - Bybit API client for market data
//! - `data` - Data loading and feature engineering
//! - `model` - Reformer model implementation
//! - `strategy` - Trading strategy and backtesting
//!
//! ## Example
//!
//! ```no_run
//! use reformer::{BybitClient, DataLoader, ReformerModel, ReformerConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Fetch cryptocurrency data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 1000).await.unwrap();
//!
//!     // Prepare dataset with features
//!     let loader = DataLoader::new();
//!     let dataset = loader.prepare_dataset(&klines, 168, 24).unwrap();
//!
//!     // Create Reformer model with LSH attention
//!     let config = ReformerConfig {
//!         seq_len: 168,
//!         d_model: 128,
//!         n_heads: 8,
//!         n_hash_rounds: 4,
//!         n_buckets: 32,
//!         ..Default::default()
//!     };
//!
//!     let model = ReformerModel::new(config);
//!
//!     // Make prediction
//!     let prediction = model.predict(&dataset.features);
//!     println!("Predicted return: {:.4}%", prediction[0] * 100.0);
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
    AttentionType, LSHAttention, OutputType, ReformerConfig, ReformerModel,
    ReversibleBlock, ChunkedFeedForward,
};
pub use strategy::{BacktestConfig, BacktestResult, SignalGenerator, TradingStrategy};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    /// Model dimension
    pub const D_MODEL: usize = 128;

    /// Number of attention heads
    pub const N_HEADS: usize = 8;

    /// Number of encoder layers
    pub const N_LAYERS: usize = 6;

    /// Number of LSH hash rounds
    pub const N_HASH_ROUNDS: usize = 4;

    /// Number of hash buckets
    pub const N_BUCKETS: usize = 32;

    /// Chunk size for attention
    pub const CHUNK_SIZE: usize = 32;

    /// Dropout probability
    pub const DROPOUT: f64 = 0.1;

    /// Sequence length (7 days of hourly data)
    pub const SEQ_LEN: usize = 168;

    /// Prediction horizon (24 hours)
    pub const PREDICTION_HORIZON: usize = 24;

    /// Learning rate
    pub const LEARNING_RATE: f64 = 0.001;

    /// Batch size
    pub const BATCH_SIZE: usize = 32;

    /// Number of epochs
    pub const EPOCHS: usize = 100;

    /// Feed-forward dimension multiplier
    pub const FF_MULT: usize = 4;
}
