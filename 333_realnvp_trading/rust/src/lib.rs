//! # RealNVP Trading
//!
//! A Rust implementation of RealNVP Normalizing Flows for cryptocurrency trading.
//!
//! This library provides:
//! - Affine coupling layers for invertible transformations
//! - RealNVP flow model with exact density estimation
//! - Bybit API integration for real-time data
//! - Trading signals based on probability density
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use realnvp_trading::{
//!     api::BybitClient,
//!     flow::RealNVP,
//!     trading::RealNVPTrader,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create RealNVP flow
//!     let flow = RealNVP::new(10, 128, 8);
//!
//!     // Use for trading
//!     let trader = RealNVPTrader::new(flow);
//!     let signal = trader.generate_signal(&candles);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod flow;
pub mod trading;
pub mod utils;

// Re-export main types
pub use api::BybitClient;
pub use backtest::Backtester;
pub use flow::{CouplingLayer, RealNVP, ActNorm};
pub use trading::{RealNVPTrader, TradingSignal};
pub use utils::{Candle, MarketState, normalize_features};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for RealNVP trading
pub mod config {
    /// Default number of coupling layers
    pub const DEFAULT_NUM_COUPLING_LAYERS: usize = 8;

    /// Default hidden dimension for coupling networks
    pub const DEFAULT_HIDDEN_DIM: usize = 128;

    /// Default number of hidden layers in coupling networks
    pub const DEFAULT_NUM_HIDDEN_LAYERS: usize = 2;

    /// Default learning rate
    pub const DEFAULT_LEARNING_RATE: f64 = 0.001;

    /// Default log probability threshold for trading
    pub const DEFAULT_LOG_PROB_THRESHOLD: f64 = -10.0;

    /// Default lookback period for features
    pub const DEFAULT_LOOKBACK: usize = 20;

    /// Default scale clamp value for numerical stability
    pub const DEFAULT_SCALE_CLAMP: f64 = 5.0;

    /// Default batch size for training
    pub const DEFAULT_BATCH_SIZE: usize = 256;
}
