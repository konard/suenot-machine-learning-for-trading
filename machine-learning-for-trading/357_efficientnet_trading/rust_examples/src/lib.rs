//! # EfficientNet Trading Library
//!
//! A modular Rust library for cryptocurrency trading using EfficientNet-based
//! image classification on price chart patterns.
//!
//! ## Features
//!
//! - **Bybit API Integration**: Fetch real-time and historical data
//! - **Image Generation**: Convert OHLCV data to chart images
//! - **EfficientNet Inference**: Pattern recognition using CNN
//! - **Trading Strategies**: Signal generation and position management
//! - **Backtesting**: Historical performance analysis
//!
//! ## Example
//!
//! ```rust,no_run
//! use efficientnet_trading::{api::BybitClient, imaging::CandlestickRenderer};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let candles = client.fetch_klines("BTCUSDT", "5", 100).await?;
//!
//!     let renderer = CandlestickRenderer::new(224, 224);
//!     let image = renderer.render(&candles);
//!
//!     image.save("btc_chart.png")?;
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod imaging;
pub mod model;
pub mod features;
pub mod strategy;
pub mod backtest;
pub mod utils;

// Re-export commonly used types
pub use api::{BybitClient, BybitWebSocket};
pub use data::{Candle, OrderBook, Trade};
pub use imaging::{CandlestickRenderer, GasfRenderer, OrderBookHeatmap};
pub use strategy::{Signal, SignalType, Position};
pub use backtest::{BacktestEngine, BacktestConfig, BacktestResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default image size for EfficientNet-B0
pub const DEFAULT_IMAGE_SIZE: u32 = 224;

/// Trading signal confidence threshold
pub const CONFIDENCE_THRESHOLD: f64 = 0.6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_IMAGE_SIZE, 224);
        assert!(CONFIDENCE_THRESHOLD > 0.5);
    }
}
