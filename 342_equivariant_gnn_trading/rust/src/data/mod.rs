//! Data Module
//!
//! Provides market data structures and Bybit API client for fetching
//! cryptocurrency data.

mod bybit_client;
mod candle;
mod orderbook;
mod dataset;

pub use bybit_client::{BybitClient, TickerInfo, FundingRate};
pub use candle::Candle;
pub use orderbook::{OrderBook, OrderBookLevel};
pub use dataset::{MarketDataset, DataPoint};
