//! # Dilated Convolutions for Trading
//!
//! This library implements dilated convolutions for cryptocurrency trading,
//! with integration to the Bybit exchange.
//!
//! ## Features
//!
//! - **Dilated Convolutions**: Multi-scale pattern recognition
//! - **WaveNet Architecture**: Gated activations and residual connections
//! - **Bybit Integration**: Real-time data fetching
//! - **Trading Strategy**: Signal generation and position sizing
//!
//! ## Example
//!
//! ```rust,no_run
//! use dilated_conv_trading::{BybitClient, DilatedConvStack, TradingStrategy};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "15", 500).await?;
//!
//!     // Create dilated convolution model
//!     let model = DilatedConvStack::new(5, 32, &[1, 2, 4, 8, 16, 32]);
//!
//!     // Generate trading signals
//!     let strategy = TradingStrategy::new(model);
//!     let signals = strategy.generate_signals(&klines);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod conv;
pub mod features;
pub mod strategy;
pub mod utils;

// Re-exports for convenient access
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker};
pub use conv::dilated::DilatedConv1D;
pub use conv::wavenet::DilatedConvStack;
pub use features::technical::TechnicalFeatures;
pub use strategy::signals::TradingStrategy;
