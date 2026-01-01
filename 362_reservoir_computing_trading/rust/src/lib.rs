//! # Reservoir Trading
//!
//! High-performance Reservoir Computing library for cryptocurrency trading.
//!
//! ## Features
//!
//! - Echo State Network (ESN) implementation with configurable parameters
//! - Online learning with Recursive Least Squares (RLS)
//! - Bybit API integration for real-time market data
//! - Feature engineering for trading signals
//! - Backtesting engine with comprehensive metrics
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use reservoir_trading::{EchoStateNetwork, EsnConfig};
//!
//! // Create an ESN with default configuration
//! let config = EsnConfig::default();
//! let mut esn = EchoStateNetwork::new(7, 1, config);
//!
//! // Train on data
//! // esn.fit(&inputs, &targets, 100);
//!
//! // Make predictions
//! // let predictions = esn.predict(&new_inputs);
//! ```

pub mod reservoir;
pub mod bybit;
pub mod trading;
pub mod features;
pub mod backtest;

// Re-export main types
pub use reservoir::{EchoStateNetwork, EsnConfig, OnlineEsn};
pub use bybit::{BybitClient, BybitConfig, Kline, OrderBook, Ticker};
pub use trading::{TradingSystem, TradingConfig, Signal, Position};
pub use features::{FeatureExtractor, MarketFeatures};
pub use backtest::{Backtester, BacktestConfig, BacktestResult, PerformanceMetrics};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default reservoir size
pub const DEFAULT_RESERVOIR_SIZE: usize = 500;

/// Default spectral radius
pub const DEFAULT_SPECTRAL_RADIUS: f64 = 0.95;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
