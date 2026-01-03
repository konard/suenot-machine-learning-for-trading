//! # Continuous Normalizing Flows Trading
//!
//! A Rust implementation of Continuous Normalizing Flows (CNF) for cryptocurrency trading.
//!
//! This library provides:
//! - Neural ODE-based continuous normalizing flows
//! - Velocity field network with time conditioning
//! - Hutchinson trace estimator for efficient log-det computation
//! - Bybit API integration for real-time data
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use cnf_trading::{
//!     api::BybitClient,
//!     cnf::ContinuousNormalizingFlow,
//!     trading::CNFTrader,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create CNF model
//!     let cnf = ContinuousNormalizingFlow::new(9, 64, 3);
//!
//!     // Use for trading
//!     let trader = CNFTrader::new(cnf);
//!     let signal = trader.generate_signal(&candles);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod cnf;
pub mod trading;
pub mod utils;

// Re-export main types
pub use api::BybitClient;
pub use backtest::Backtester;
pub use cnf::{ContinuousNormalizingFlow, VelocityField, ODESolver};
pub use trading::{CNFTrader, TradingSignal};
pub use utils::{Candle, MarketState, compute_market_features, normalize_features};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for CNF
pub mod config {
    /// Default hidden dimension for velocity field
    pub const DEFAULT_HIDDEN_DIM: usize = 64;

    /// Default number of layers in velocity field
    pub const DEFAULT_NUM_LAYERS: usize = 3;

    /// Default number of ODE integration steps
    pub const DEFAULT_ODE_STEPS: usize = 50;

    /// Default time span for flow [0, T]
    pub const DEFAULT_T_SPAN: (f64, f64) = (0.0, 1.0);

    /// Default learning rate
    pub const DEFAULT_LEARNING_RATE: f64 = 0.001;

    /// Default likelihood threshold for trading
    pub const DEFAULT_LIKELIHOOD_THRESHOLD: f64 = -10.0;

    /// Default confidence threshold for trading
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.6;

    /// Default lookback period for features
    pub const DEFAULT_LOOKBACK: usize = 20;

    /// Feature dimension (number of market features)
    pub const FEATURE_DIM: usize = 9;
}
