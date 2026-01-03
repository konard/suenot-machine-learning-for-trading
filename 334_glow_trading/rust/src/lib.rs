//! GLOW Trading - Generative Flow with Invertible 1x1 Convolutions for Cryptocurrency Trading
//!
//! This library implements the GLOW model for trading applications, featuring:
//! - Exact likelihood computation for anomaly detection
//! - Invertible transformations for both encoding and generation
//! - Multi-scale architecture for capturing market patterns
//! - Scenario generation for risk management
//!
//! # Example
//!
//! ```no_run
//! use glow_trading::{GLOWModel, GLOWConfig, BybitClient, Interval};
//! use chrono::{Utc, Duration};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client
//!         .get_klines("BTCUSDT", Interval::OneHour,
//!                     Utc::now() - Duration::days(30), Utc::now())
//!         .await?;
//!
//!     // Create and train GLOW model
//!     let config = GLOWConfig::default();
//!     let model = GLOWModel::new(config);
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod model;
pub mod trading;
pub mod utils;

// Re-export main types
pub use data::{BybitClient, Candle, Interval, Ticker};
pub use model::{GLOWModel, GLOWConfig, FlowStep, ActNorm, InvertibleConv1x1, AffineCoupling};
pub use trading::{GLOWTrader, TraderConfig, BacktestResult, TradingSignal};
pub use utils::{Config, Checkpoint};
