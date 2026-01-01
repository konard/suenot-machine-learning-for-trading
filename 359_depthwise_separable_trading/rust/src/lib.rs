//! # Depthwise Separable Trading
//!
//! High-performance depthwise separable convolutions for cryptocurrency trading.
//!
//! This library provides:
//! - Efficient 1D convolution operations (depthwise, pointwise, separable)
//! - Bybit exchange data integration
//! - Trading strategy framework
//! - Technical indicators
//!
//! ## Example
//!
//! ```rust,no_run
//! use dsc_trading::{
//!     convolution::DepthwiseSeparableConv1d,
//!     data::BybitClient,
//!     strategy::TradingStrategy,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create model
//!     let model = DepthwiseSeparableConv1d::new(10, 64, 3)?;
//!
//!     // Run strategy
//!     let strategy = TradingStrategy::new(model);
//!     let signals = strategy.generate_signals(&candles)?;
//!
//!     Ok(())
//! }
//! ```

pub mod convolution;
pub mod data;
pub mod indicators;
pub mod strategy;

// Re-exports for convenience
pub use convolution::{
    DepthwiseConv1d, DepthwiseSeparableConv1d, PointwiseConv1d,
};
pub use data::{BybitClient, Candle};
pub use indicators::TechnicalIndicators;
pub use strategy::{Backtest, Signal, TradingStrategy};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::convolution::*;
    pub use crate::data::*;
    pub use crate::indicators::*;
    pub use crate::strategy::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
