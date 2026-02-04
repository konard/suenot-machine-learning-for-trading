//! # EfficientNet Trading
//!
//! This library provides implementations for EfficientNet-based trading strategies
//! using time series to image transformations for cryptocurrency trading.
//!
//! ## Core Concepts
//!
//! - **Image Transformation**: Convert OHLCV data to GAF/MTF images
//! - **EfficientNet**: Efficient CNN architecture for pattern recognition
//! - **Trading Signals**: Generate signals from image-based predictions
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `image` - Time series to image transformations (GAF, MTF)
//! - `model` - EfficientNet model structure and inference
//! - `strategy` - Trading signal generation
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use efficientnet_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 120).await?;
//!
//!     // Convert to GAF image
//!     let transformer = GAFTransformer::new(224);
//!     let image = transformer.transform(&klines.close_prices());
//!
//!     // Generate signal (placeholder - actual model inference not shown)
//!     let signal = Signal::new(SignalType::Hold, 0.0);
//!
//!     println!("{:?}", signal);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod image;
pub mod model;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, Ticker};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use image::gaf::GAFTransformer;
pub use image::mtf::MTFTransformer;
pub use image::stack::ImageStackGenerator;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols for examples
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
];

/// EfficientNet variant configurations
#[derive(Debug, Clone, Copy)]
pub struct EfficientNetConfig {
    pub variant: EfficientNetVariant,
    pub image_size: usize,
    pub num_classes: usize,
}

/// EfficientNet model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfficientNetVariant {
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
}

impl EfficientNetVariant {
    /// Get the default image size for this variant
    pub fn image_size(&self) -> usize {
        match self {
            Self::B0 => 224,
            Self::B1 => 240,
            Self::B2 => 260,
            Self::B3 => 300,
            Self::B4 => 380,
            Self::B5 => 456,
            Self::B6 => 528,
            Self::B7 => 600,
        }
    }

    /// Get approximate parameter count (millions)
    pub fn params_millions(&self) -> f32 {
        match self {
            Self::B0 => 5.3,
            Self::B1 => 7.8,
            Self::B2 => 9.2,
            Self::B3 => 12.0,
            Self::B4 => 19.0,
            Self::B5 => 30.0,
            Self::B6 => 43.0,
            Self::B7 => 66.0,
        }
    }

    /// Get ImageNet Top-1 accuracy
    pub fn imagenet_accuracy(&self) -> f32 {
        match self {
            Self::B0 => 77.1,
            Self::B1 => 79.1,
            Self::B2 => 80.1,
            Self::B3 => 81.6,
            Self::B4 => 82.9,
            Self::B5 => 83.6,
            Self::B6 => 84.0,
            Self::B7 => 84.3,
        }
    }
}

impl Default for EfficientNetConfig {
    fn default() -> Self {
        Self {
            variant: EfficientNetVariant::B2,
            image_size: 260,
            num_classes: 3,
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::client::BybitClient;
    pub use crate::api::types::{Kline, Ticker};
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::backtest::report::BacktestReport;
    pub use crate::image::gaf::GAFTransformer;
    pub use crate::image::mtf::MTFTransformer;
    pub use crate::image::stack::ImageStackGenerator;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::{EfficientNetConfig, EfficientNetVariant, DEFAULT_SYMBOLS};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_image_sizes() {
        assert_eq!(EfficientNetVariant::B0.image_size(), 224);
        assert_eq!(EfficientNetVariant::B2.image_size(), 260);
        assert_eq!(EfficientNetVariant::B7.image_size(), 600);
    }

    #[test]
    fn test_default_config() {
        let config = EfficientNetConfig::default();
        assert_eq!(config.variant, EfficientNetVariant::B2);
        assert_eq!(config.image_size, 260);
        assert_eq!(config.num_classes, 3);
    }
}
