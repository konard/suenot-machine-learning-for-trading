//! # Order Flow Imbalance for Cryptocurrency Trading
//!
//! This library provides implementations for analyzing order flow imbalance
//! in cryptocurrency markets using data from Bybit exchange.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching order book and trade data
//! - `data` - Data structures and processing utilities
//! - `orderflow` - OFI, VPIN, and other microstructure metrics
//! - `features` - Feature engineering from order book snapshots
//! - `models` - Machine learning models for direction prediction
//! - `metrics` - Model evaluation and trading performance metrics
//! - `strategy` - Trading signal generation and execution
//! - `backtest` - Backtesting framework for intraday strategies
//!
//! ## Example
//!
//! ```rust,no_run
//! use order_flow_imbalance::{BybitClient, OrderFlowCalculator, FeatureEngine};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch order book data
//!     let client = BybitClient::new();
//!     let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
//!
//!     // Calculate OFI
//!     let calculator = OrderFlowCalculator::new();
//!     let ofi = calculator.calculate_ofi(&orderbook);
//!
//!     println!("Current OFI: {:.4}", ofi);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod data;
pub mod features;
pub mod metrics;
pub mod models;
pub mod orderflow;
pub mod strategy;

// Re-export commonly used types
pub use api::bybit::BybitClient;
pub use api::websocket::BybitWebSocket;
pub use backtest::engine::BacktestEngine;
pub use data::orderbook::{OrderBook, OrderBookLevel};
pub use data::trade::Trade;
pub use features::engine::FeatureEngine;
pub use metrics::classification::ClassificationMetrics;
pub use metrics::trading::TradingMetrics;
pub use models::gradient_boosting::GradientBoostingModel;
pub use orderflow::ofi::OrderFlowCalculator;
pub use orderflow::vpin::VpinCalculator;
pub use strategy::signal::SignalGenerator;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default symbol for examples
pub const DEFAULT_SYMBOL: &str = "BTCUSDT";

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::bybit::BybitClient;
    pub use crate::api::websocket::BybitWebSocket;
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::data::orderbook::{OrderBook, OrderBookLevel};
    pub use crate::data::trade::Trade;
    pub use crate::features::engine::FeatureEngine;
    pub use crate::metrics::classification::ClassificationMetrics;
    pub use crate::metrics::trading::TradingMetrics;
    pub use crate::models::gradient_boosting::GradientBoostingModel;
    pub use crate::orderflow::ofi::OrderFlowCalculator;
    pub use crate::orderflow::vpin::VpinCalculator;
    pub use crate::strategy::signal::SignalGenerator;
}
