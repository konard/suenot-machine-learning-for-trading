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
//! use candle_core::{DType, Device};
//! use candle_nn::{VarBuilder, VarMap};
//! use cross_attention_multi_asset::{
//!     model::{CrossAttentionMultiAsset, ModelConfig, OutputType},
//!     data::{BybitClient, compute_features},
//!     strategy::{Backtest, BacktestConfig},
//! };
//!
//! // Create model with VarBuilder
//! let device = Device::Cpu;
//! let varmap = VarMap::new();
//! let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
//! let config = ModelConfig::default();
//! let model = CrossAttentionMultiAsset::new(&config, vb)?;
//!
//! // Fetch data
//! let client = BybitClient::new();
//! let candles = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//! // Run backtest with pre-computed weights and returns
//! let backtest = Backtest::new(BacktestConfig::default());
//! let weights: Vec<Vec<f64>> = vec![vec![0.2; 5]; 100];  // Example weights
//! let returns: Vec<Vec<f64>> = vec![vec![0.01; 5]; 100]; // Example returns
//! let symbols = vec!["BTC".to_string(), "ETH".to_string()];
//! let timestamps: Vec<i64> = (0..100).map(|i| i * 3600000).collect();
//! let results = backtest.run(&weights, &returns, &symbols, &timestamps);
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
