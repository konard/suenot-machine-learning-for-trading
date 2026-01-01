//! # ConvNeXt Trading Library
//!
//! A Rust implementation of ConvNeXt architecture for cryptocurrency trading.
//! Includes Bybit data fetching, feature engineering, and trading signal generation.
//!
//! ## Modules
//!
//! - `convnext` - ConvNeXt neural network architecture
//! - `data` - Data fetching and processing (Bybit integration)
//! - `trading` - Trading signals and backtesting
//! - `utils` - Utility functions and metrics

pub mod convnext;
pub mod data;
pub mod trading;
pub mod utils;

pub use convnext::{ConvNeXt, ConvNeXtConfig, ConvNeXtBlock};
pub use data::{BybitClient, Candle, Dataset, FeatureBuilder};
pub use trading::{Signal, Strategy, Backtest, BacktestMetrics};
pub use utils::Metrics;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::convnext::*;
    pub use crate::data::*;
    pub use crate::trading::*;
    pub use crate::utils::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
