//! API data types for Bybit exchange

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub start_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl Kline {
    /// Calculate the typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if the candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Order book price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: i64,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
}

impl OrderBook {
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            Some((best_bid.price + best_ask.price) / 2.0)
        } else {
            None
        }
    }

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            let mid = (best_bid.price + best_ask.price) / 2.0;
            Some((best_ask.price - best_bid.price) / mid * 10000.0)
        } else {
            None
        }
    }

    /// Calculate depth imbalance at given levels
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid_depth: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();
        let ask_depth: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();
        let total = bid_depth + ask_depth;
        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub price_change_24h: f64,
    pub timestamp: i64,
}

impl Ticker {
    /// Calculate current spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> f64 {
        let mid = (self.bid_price + self.ask_price) / 2.0;
        if mid > 0.0 {
            (self.ask_price - self.bid_price) / mid * 10000.0
        } else {
            0.0
        }
    }
}

/// Individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub timestamp: i64,
}

impl Trade {
    /// Check if trade is a buy
    pub fn is_buy(&self) -> bool {
        self.side.to_lowercase() == "buy"
    }

    /// Calculate trade value
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: T,
    pub time: i64,
}

/// Kline API result
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Ticker API result
#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub category: String,
    pub list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "bid1Price")]
    pub bid_price: String,
    #[serde(rename = "ask1Price")]
    pub ask_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_change_24h: String,
}

/// Order book API result
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String,  // symbol
    pub b: Vec<Vec<String>>,  // bids
    pub a: Vec<Vec<String>>,  // asks
    pub ts: i64,  // timestamp
    pub u: i64,   // update id
}
