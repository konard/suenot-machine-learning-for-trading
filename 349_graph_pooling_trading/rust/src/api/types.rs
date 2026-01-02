//! API types for Bybit exchange

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// API error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Rate limit exceeded")]
    RateLimited,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("WebSocket error: {0}")]
    WebSocketError(String),
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,

    /// Opening time in milliseconds
    pub open_time: i64,

    /// Opening price
    pub open: f64,

    /// Highest price
    pub high: f64,

    /// Lowest price
    pub low: f64,

    /// Closing price
    pub close: f64,

    /// Trading volume
    pub volume: f64,

    /// Turnover (quote asset volume)
    pub turnover: f64,
}

impl Kline {
    /// Calculate the return (close / open - 1)
    pub fn return_rate(&self) -> f64 {
        self.close / self.open - 1.0
    }

    /// Calculate the log return
    pub fn log_return(&self) -> f64 {
        (self.close / self.open).ln()
    }

    /// Calculate the range (high - low) / open
    pub fn range(&self) -> f64 {
        (self.high - self.low) / self.open
    }

    /// Calculate typical price (H + L + C) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Order book level (price + quantity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,

    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,

    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,

    /// Timestamp in milliseconds
    pub timestamp: i64,
}

impl OrderBook {
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate bid depth (sum of bid quantities up to n levels)
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.quantity).sum()
    }

    /// Calculate ask depth (sum of ask quantities up to n levels)
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.quantity).sum()
    }

    /// Calculate order imbalance
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_depth = self.bid_depth(levels);
        let ask_depth = self.ask_depth(levels);
        let total = bid_depth + ask_depth;
        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }
}

/// Ticker data (24h statistics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,

    /// Last traded price
    pub last_price: f64,

    /// 24h high
    pub high_24h: f64,

    /// 24h low
    pub low_24h: f64,

    /// 24h volume
    pub volume_24h: f64,

    /// 24h turnover
    pub turnover_24h: f64,

    /// 24h price change percentage
    pub price_change_24h: f64,

    /// Bid price
    pub bid_price: f64,

    /// Ask price
    pub ask_price: f64,
}

impl Ticker {
    /// Calculate 24h volatility estimate
    pub fn volatility_24h(&self) -> f64 {
        if self.last_price > 0.0 {
            (self.high_24h - self.low_24h) / self.last_price
        } else {
            0.0
        }
    }
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Symbol
    pub symbol: String,

    /// Trade ID
    pub trade_id: String,

    /// Trade price
    pub price: f64,

    /// Trade quantity
    pub quantity: f64,

    /// Trade timestamp in milliseconds
    pub timestamp: i64,

    /// Is buyer the maker
    pub is_buyer_maker: bool,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,

    #[serde(rename = "retMsg")]
    pub ret_msg: String,

    pub result: Option<T>,

    pub time: Option<i64>,
}

/// Klines response
#[derive(Debug, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Tickers response
#[derive(Debug, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
pub struct TickerItem {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "bid1Price")]
    pub bid1_price: String,
    #[serde(rename = "ask1Price")]
    pub ask1_price: String,
}

/// Order book response
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String, // symbol
    pub b: Vec<Vec<String>>, // bids
    pub a: Vec<Vec<String>>, // asks
    pub ts: i64, // timestamp
    pub u: i64, // update id
}
