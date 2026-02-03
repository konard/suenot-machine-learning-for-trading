//! # Mamba4Cast Zero-Shot Forecasting
//!
//! A Rust implementation of Mamba4Cast for zero-shot time series forecasting.
//!
//! This crate provides high-performance inference for Mamba4Cast models,
//! enabling real-time zero-shot forecasting for trading applications.
//!
//! ## Features
//!
//! - Zero-shot time series forecasting
//! - Non-autoregressive multi-horizon prediction
//! - Integration with Bybit cryptocurrency exchange
//! - High-performance inference
//!
//! ## Example
//!
//! ```rust,ignore
//! use mamba4cast::prelude::*;
//!
//! let forecaster = Mamba4CastForecaster::new(64, 4, 16, 96);
//! let prices: Vec<f32> = vec![100.0, 101.0, 102.0, /* ... */];
//! let forecast = forecaster.zero_shot_forecast(&prices, 100, 24);
//! ```

pub mod model;
pub mod data;

pub mod prelude {
    //! Convenience re-exports for common usage.
    pub use crate::model::{Mamba4CastBlock, Mamba4CastForecaster, TradingSignal};
    pub use crate::data::{BybitClient, StockData, KlineData};
}

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
