//! # FNet Trading Library
//!
//! Implementation of FNet (Fourier Transform) architecture for financial time series prediction.
//!
//! ## Features
//!
//! - **O(n log n) complexity**: Replaces attention with FFT for efficient token mixing
//! - **Bybit API integration**: Fetch cryptocurrency data
//! - **Backtesting framework**: Evaluate trading strategies
//!
//! ## Example
//!
//! ```rust,ignore
//! use fnet_trading::{FNet, BybitClient, Backtester};
//!
//! // Fetch data
//! let client = BybitClient::new();
//! let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//! // Create model
//! let model = FNet::new(7, 256, 4);
//!
//! // Run backtest
//! let backtester = Backtester::new(100000.0);
//! let results = backtester.run(&strategy, &data);
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-export main types
pub use api::client::BybitClient;
pub use api::types::Kline;
pub use data::dataset::TradingDataset;
pub use data::features::{calculate_features, FeatureConfig, TradingFeatures};
pub use model::encoder::FNetEncoderBlock;
pub use model::fnet::FNet;
pub use model::fourier::FourierLayer;
pub use strategy::backtest::{BacktestResult, Backtester, BacktesterConfig, TradeMetrics};
pub use strategy::signals::{Signal, SignalGenerator, SignalGeneratorConfig, TradingSignal};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model configuration
pub struct FNetConfig {
    pub n_features: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub dropout: f64,
    pub max_seq_len: usize,
}

impl Default for FNetConfig {
    fn default() -> Self {
        Self {
            n_features: 7,
            d_model: 256,
            n_layers: 4,
            d_ff: 1024,
            dropout: 0.1,
            max_seq_len: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FNetConfig::default();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_layers, 4);
    }
}
