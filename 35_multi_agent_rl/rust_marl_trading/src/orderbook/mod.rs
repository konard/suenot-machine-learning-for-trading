//! Order book simulation module
//!
//! Provides a realistic order book implementation with price-time priority matching.

mod order;
mod book;
mod matching;

pub use order::{Order, OrderId, OrderSide, OrderType};
pub use book::OrderBook;
pub use matching::{Trade, MatchResult};
