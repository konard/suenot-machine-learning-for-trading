//! Market data structures and exchange connectivity.
//!
//! This module provides:
//! - Common data types for market data (OHLCV, OrderBook, Trades)
//! - Bybit exchange client for cryptocurrency data
//! - Real-time WebSocket data streaming

mod bybit;
mod orderbook;
mod types;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use orderbook::{OrderBook, OrderBookLevel, OrderBookSnapshot};
pub use types::{
    MarketData, MarketDataError, OhlcvBar, PriceData, Ticker, TimeFrame, Trade,
    TradeDirection, Volume,
};
