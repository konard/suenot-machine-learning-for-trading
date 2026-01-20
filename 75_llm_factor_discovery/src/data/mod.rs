//! Market data module
//!
//! Provides data structures and clients for fetching market data
//! from various sources including Bybit exchange.

mod bybit;
mod buffer;
mod types;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use buffer::DataBuffer;
pub use types::{MarketData, MarketDataError, OhlcvBar, OrderBook, PriceData, Ticker, TimeFrame};
