//! Data structures for market data
//!
//! Core types for candlesticks, order books, and trades.

mod candle;
mod orderbook;

pub use candle::Candle;
pub use orderbook::{OrderBook, OrderBookLevel, OrderBookSnapshot};

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Trade data
#[derive(Debug, Clone)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub timestamp: u64,
}

impl Trade {
    /// Get notional value
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }

    /// Check if this is a buy trade
    pub fn is_buy(&self) -> bool {
        self.side == TradeSide::Buy
    }
}
