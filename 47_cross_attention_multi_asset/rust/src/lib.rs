//! Cross-Attention Multi-Asset Trading
//!
//! High-performance Rust implementation of Cross-Attention mechanism
//! for multi-asset trading using the Candle ML framework.
//!
//! # Features
//!
//! - **Cross-Attention Model**: Transformer-based model with cross-asset attention
//! - **Bybit Integration**: Fetch cryptocurrency data from Bybit exchange
//! - **Feature Engineering**: Technical indicators and feature computation
//! - **Backtesting Engine**: Evaluate strategies with realistic constraints
//!
//! # Example
//!
//! ```rust,ignore
//! use cross_attention_multi_asset::{
//!     model::{CrossAttentionMultiAsset, ModelConfig, OutputType},
//!     data::{BybitClient, compute_features},
//!     strategy::{Backtest, BacktestConfig},
//! };
//!
//! // Create model
//! let config = ModelConfig::default();
//! let model = CrossAttentionMultiAsset::new(&config)?;
//!
//! // Fetch data
//! let client = BybitClient::new();
//! let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//! // Run backtest
//! let backtest_config = BacktestConfig::default();
//! let results = Backtest::run(&model, &data, &backtest_config)?;
//! ```

pub mod model;
pub mod data;
pub mod strategy;

// Re-export main types
pub use model::{CrossAttentionMultiAsset, ModelConfig, OutputType};
pub use data::{BybitClient, Candle, compute_features};
pub use strategy::{Backtest, BacktestConfig, BacktestResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model dimension
pub const DEFAULT_D_MODEL: usize = 64;

/// Default number of attention heads
pub const DEFAULT_N_HEADS: usize = 4;

/// Default sequence length (7 days of hourly data)
pub const DEFAULT_SEQ_LEN: usize = 168;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_defaults() {
        assert_eq!(DEFAULT_D_MODEL, 64);
        assert_eq!(DEFAULT_N_HEADS, 4);
        assert_eq!(DEFAULT_SEQ_LEN, 168);
    }
}
