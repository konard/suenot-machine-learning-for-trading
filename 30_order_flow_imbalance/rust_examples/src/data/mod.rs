//! # Data Module
//!
//! Data structures and processing utilities for order flow analysis.
//!
//! ## Modules
//!
//! - `orderbook` - Order book data structures
//! - `trade` - Trade data structures
//! - `snapshot` - Time-series snapshots for analysis

pub mod orderbook;
pub mod snapshot;
pub mod trade;

pub use orderbook::{OrderBook, OrderBookLevel};
pub use snapshot::MarketSnapshot;
pub use trade::Trade;
