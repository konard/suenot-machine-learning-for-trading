//! # DenseNet Trading Library
//!
//! A Rust implementation of DenseNet (Densely Connected Convolutional Networks)
//! for cryptocurrency trading on Bybit exchange.
//!
//! ## Architecture
//!
//! DenseNet's key innovation is dense connectivity: each layer receives feature maps
//! from ALL preceding layers. This enables:
//! - Excellent feature reuse
//! - Strong gradient flow
//! - Parameter efficiency
//!
//! ## Modules
//!
//! - `densenet`: Core DenseNet architecture implementation
//! - `data`: Bybit API client and data handling
//! - `features`: Technical indicators and feature extraction
//! - `trading`: Trading signals, backtesting, and risk management
//! - `utils`: Helper functions and common utilities

pub mod densenet;
pub mod data;
pub mod features;
pub mod trading;
pub mod utils;

// Re-export commonly used types
pub use densenet::{DenseNet, DenseBlock, DenseLayer, TransitionLayer, DenseNetConfig};
pub use data::{BybitClient, Candle, OrderBook};
pub use features::{FeatureExtractor, TechnicalIndicators};
pub use trading::{TradingSignal, Position, Backtester, RiskManager};
pub use utils::{normalize, standardize};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default growth rate for DenseNet layers
pub const DEFAULT_GROWTH_RATE: usize = 32;

/// Default compression factor for transition layers
pub const DEFAULT_COMPRESSION: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_GROWTH_RATE, 32);
        assert!((DEFAULT_COMPRESSION - 0.5).abs() < 1e-10);
    }
}
