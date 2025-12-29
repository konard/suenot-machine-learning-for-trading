//! # Rust Backtester
//!
//! A modular cryptocurrency backtesting framework with Bybit integration.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `models` - Data structures for candles, orders, positions
//! - `backtest` - Backtesting engine
//! - `strategies` - Trading strategy traits and implementations
//! - `utils` - Utility functions and indicators

pub mod api;
pub mod backtest;
pub mod models;
pub mod strategies;
pub mod utils;

pub use api::BybitClient;
pub use backtest::{BacktestEngine, BacktestResult};
pub use models::{Candle, Order, OrderSide, OrderType, Position, Timeframe};
pub use strategies::{Signal, Strategy};
