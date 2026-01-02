//! Data handling module for fetching and processing market data.

mod bybit;

pub use bybit::*;

use serde::{Deserialize, Serialize};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Symbol
    pub symbol: String,
}

impl Candle {
    /// Create a new candle.
    pub fn new(
        timestamp: u64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        symbol: impl Into<String>,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            symbol: symbol.into(),
        }
    }

    /// Get the candle's price range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get the candle body size.
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if the candle is bullish.
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get the typical price.
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Order book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Timestamp
    pub timestamp: u64,
    /// Symbol
    pub symbol: String,
    /// Bid prices and quantities
    pub bids: Vec<(f64, f64)>,
    /// Ask prices and quantities
    pub asks: Vec<(f64, f64)>,
}

impl OrderBook {
    /// Get the best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
    }

    /// Get the mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get the spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get the spread in basis points.
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Get total bid volume up to a depth.
    pub fn bid_volume(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|(_, q)| q).sum()
    }

    /// Get total ask volume up to a depth.
    pub fn ask_volume(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|(_, q)| q).sum()
    }

    /// Get order imbalance.
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_vol = self.bid_volume(depth);
        let ask_vol = self.ask_volume(depth);
        let total = bid_vol + ask_vol;

        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

/// Trade data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Timestamp
    pub timestamp: u64,
    /// Symbol
    pub symbol: String,
    /// Price
    pub price: f64,
    /// Quantity
    pub quantity: f64,
    /// Whether this was a buy (taker was buyer)
    pub is_buyer_maker: bool,
}

/// Market data aggregator.
pub struct MarketData {
    /// Candle data by symbol
    pub candles: std::collections::HashMap<String, Vec<Candle>>,
    /// Latest order book by symbol
    pub orderbooks: std::collections::HashMap<String, OrderBook>,
    /// Recent trades by symbol
    pub trades: std::collections::HashMap<String, Vec<Trade>>,
}

impl MarketData {
    /// Create a new market data aggregator.
    pub fn new() -> Self {
        Self {
            candles: std::collections::HashMap::new(),
            orderbooks: std::collections::HashMap::new(),
            trades: std::collections::HashMap::new(),
        }
    }

    /// Add candles for a symbol.
    pub fn add_candles(&mut self, symbol: impl Into<String>, candles: Vec<Candle>) {
        self.candles.insert(symbol.into(), candles);
    }

    /// Update order book for a symbol.
    pub fn update_orderbook(&mut self, orderbook: OrderBook) {
        self.orderbooks.insert(orderbook.symbol.clone(), orderbook);
    }

    /// Add trades for a symbol.
    pub fn add_trades(&mut self, symbol: impl Into<String>, trades: Vec<Trade>) {
        self.trades.insert(symbol.into(), trades);
    }

    /// Get all symbols.
    pub fn symbols(&self) -> Vec<&str> {
        self.candles.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for MarketData {
    fn default() -> Self {
        Self::new()
    }
}
