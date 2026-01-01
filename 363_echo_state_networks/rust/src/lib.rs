//! # ESN Trading Library
//!
//! Echo State Networks for Cryptocurrency Trading on Bybit.
//!
//! This library provides a complete implementation of Echo State Networks (ESN)
//! for time series prediction and trading signal generation, with integration
//! to the Bybit cryptocurrency exchange.
//!
//! ## Features
//!
//! - **ESN Core**: Complete implementation of Echo State Networks with configurable
//!   reservoir size, spectral radius, leaking rate, and regularization
//! - **Bybit API**: Async client for fetching historical and real-time market data
//! - **Trading**: Feature engineering, signal generation, and backtesting framework
//! - **Utils**: Performance metrics and evaluation tools
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use esn_trading::{EchoStateNetwork, ESNConfig, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 1000, None).await?;
//!
//!     // Configure and create ESN
//!     let config = ESNConfig::default()
//!         .reservoir_size(500)
//!         .spectral_radius(0.95);
//!
//!     let mut esn = EchoStateNetwork::new(config);
//!
//!     // Train and predict
//!     // ...
//!
//!     Ok(())
//! }
//! ```

pub mod esn;
pub mod api;
pub mod trading;
pub mod utils;

// Re-export main types for convenience
pub use esn::{EchoStateNetwork, ESNConfig, DeepESN, EnsembleESN};
pub use api::{BybitClient, Kline, OrderBook};
pub use trading::{FeatureEngineering, TradingSignal, Backtest, BacktestConfig, BacktestResult};
pub use utils::{PerformanceMetrics, PredictionMetrics};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration presets
pub mod presets {
    use crate::ESNConfig;

    /// Configuration optimized for high-frequency trading (1-minute candles)
    pub fn hft() -> ESNConfig {
        ESNConfig::default()
            .reservoir_size(200)
            .spectral_radius(0.9)
            .leaking_rate(0.5)
            .input_scaling(0.1)
            .regularization(1e-6)
    }

    /// Configuration for swing trading (hourly candles)
    pub fn swing() -> ESNConfig {
        ESNConfig::default()
            .reservoir_size(500)
            .spectral_radius(0.95)
            .leaking_rate(0.3)
            .input_scaling(0.1)
            .regularization(1e-6)
    }

    /// Configuration for position trading (daily candles)
    pub fn position() -> ESNConfig {
        ESNConfig::default()
            .reservoir_size(1000)
            .spectral_radius(0.99)
            .leaking_rate(0.1)
            .input_scaling(0.05)
            .regularization(1e-8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_presets() {
        let hft = presets::hft();
        assert_eq!(hft.reservoir_size, 200);

        let swing = presets::swing();
        assert_eq!(swing.reservoir_size, 500);

        let position = presets::position();
        assert_eq!(position.reservoir_size, 1000);
    }
}
