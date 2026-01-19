//! BigBird Trading - Rust Implementation
//!
//! High-performance implementation of BigBird sparse attention for financial time series prediction.
//!
//! # Features
//!
//! - **BigBird Sparse Attention**: Efficient O(n) attention mechanism
//!   - Window (local) attention
//!   - Random attention connections
//!   - Global tokens
//! - **Bybit API Integration**: Real-time cryptocurrency data
//! - **Stock Data Support**: Yahoo Finance compatible
//! - **Backtesting Engine**: Complete strategy evaluation
//!
//! # Example
//!
//! ```no_run
//! use bigbird_trading::{BigBirdConfig, BigBirdModel};
//!
//! let config = BigBirdConfig::default();
//! let model = BigBirdModel::new(&config);
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-export main types for convenience
pub use api::{BybitClient, KlineData};
pub use data::{DataLoader, FeatureEngine, TradingDataset};
pub use model::{BigBirdConfig, BigBirdModel, BigBirdSparseAttention};
pub use strategy::{BacktestConfig, BacktestResult, SignalGenerator};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sequence length for trading models
pub const DEFAULT_SEQ_LEN: usize = 256;

/// Default model dimension
pub const DEFAULT_D_MODEL: usize = 128;

/// Default number of attention heads
pub const DEFAULT_N_HEADS: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BigBirdConfig::default();
        assert_eq!(config.seq_len, DEFAULT_SEQ_LEN);
        assert_eq!(config.d_model, DEFAULT_D_MODEL);
        assert_eq!(config.n_heads, DEFAULT_N_HEADS);
    }
}
