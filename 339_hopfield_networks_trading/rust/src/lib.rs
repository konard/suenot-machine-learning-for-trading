//! # Hopfield Trading
//!
//! A Rust library implementing Hopfield Networks for cryptocurrency trading
//! pattern recognition using data from the Bybit exchange.
//!
//! ## Features
//!
//! - Classical and Modern Hopfield Network implementations
//! - Bybit API client for real-time and historical data
//! - Pattern encoding for OHLCV data
//! - Trading signal generation based on pattern matching
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use hopfield_trading::{HopfieldNetwork, BybitClient, PatternEncoder};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize client and fetch data
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 100).await?;
//!
//!     // Encode patterns
//!     let encoder = PatternEncoder::default();
//!     let patterns = encoder.encode_candles(&candles);
//!
//!     // Create and train network
//!     let mut network = HopfieldNetwork::new(patterns[0].len());
//!     network.store_patterns(&patterns);
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod hopfield;
pub mod trading;
pub mod utils;

// Re-export main types for convenience
pub use data::bybit::BybitClient;
pub use data::Candle;
pub use hopfield::classical::ClassicalHopfield;
pub use hopfield::modern::ModernHopfield;
pub use hopfield::HopfieldNetwork;
pub use trading::patterns::PatternEncoder;
pub use trading::signals::{Signal, SignalGenerator};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::Candle;
    pub use crate::hopfield::classical::ClassicalHopfield;
    pub use crate::hopfield::modern::ModernHopfield;
    pub use crate::hopfield::HopfieldNetwork;
    pub use crate::trading::patterns::PatternEncoder;
    pub use crate::trading::signals::{Signal, SignalGenerator};
}
