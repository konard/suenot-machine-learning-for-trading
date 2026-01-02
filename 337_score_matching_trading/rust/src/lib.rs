//! # Score Matching Trading
//!
//! A Rust implementation of Score Matching for cryptocurrency trading.
//!
//! This library provides:
//! - Score network implementation for learning data distributions
//! - Denoising Score Matching training
//! - Langevin dynamics for sampling and prediction
//! - Bybit API integration for real-time data
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use score_matching_trading::{
//!     api::BybitClient,
//!     score::ScoreNetwork,
//!     trading::ScoreMatchingTrader,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create and train score network
//!     let network = ScoreNetwork::new(10, 128, 4);
//!
//!     // Use for trading
//!     let trader = ScoreMatchingTrader::new(network);
//!     let signal = trader.generate_signal(&candles);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod score;
pub mod trading;
pub mod utils;

// Re-export main types
pub use api::BybitClient;
pub use backtest::Backtester;
pub use score::{ScoreNetwork, MultiScaleScoreNetwork, DenoisingScoreMatchingTrainer};
pub use trading::{ScoreMatchingTrader, TradingSignal};
pub use utils::{Candle, MarketState, normalize_features};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for score matching
pub mod config {
    /// Default number of noise levels for multi-scale training
    pub const DEFAULT_NOISE_LEVELS: usize = 10;

    /// Default minimum noise sigma
    pub const DEFAULT_SIGMA_MIN: f64 = 0.01;

    /// Default maximum noise sigma
    pub const DEFAULT_SIGMA_MAX: f64 = 1.0;

    /// Default learning rate
    pub const DEFAULT_LEARNING_RATE: f64 = 0.001;

    /// Default hidden dimension for score network
    pub const DEFAULT_HIDDEN_DIM: usize = 128;

    /// Default number of layers
    pub const DEFAULT_NUM_LAYERS: usize = 4;

    /// Default confidence threshold for trading
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.3;

    /// Default lookback period for features
    pub const DEFAULT_LOOKBACK: usize = 20;
}
