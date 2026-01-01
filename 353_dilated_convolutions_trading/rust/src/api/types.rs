//! API response types

use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Trading symbol
    pub symbol: String,
    /// Interval (e.g., "15", "60", "D")
    pub interval: String,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Create a new Kline
    pub fn new(
        timestamp: i64,
        symbol: impl Into<String>,
        interval: impl Into<String>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            symbol: symbol.into(),
            interval: interval.into(),
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Calculate the price return
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate the high-low range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the range as percentage of close
    pub fn range_pct(&self) -> f64 {
        if self.close > 0.0 {
            (self.high - self.low) / self.close
        } else {
            0.0
        }
    }

    /// Get close position within the bar (0 to 1)
    pub fn close_position(&self) -> f64 {
        let range = self.high - self.low;
        if range > 0.0 {
            (self.close - self.low) / range
        } else {
            0.5
        }
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Size at this level
    pub size: f64,
}

impl OrderBookLevel {
    /// Create a new level
    pub fn new(price: f64, size: f64) -> Self {
        Self { price, size }
    }
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,
    /// Bid levels (buy orders)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sell orders)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: i64,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(
        symbol: impl Into<String>,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
        timestamp: i64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            bids,
            asks,
            timestamp,
        }
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get the spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get the spread as percentage of mid price
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid),
            _ => None,
        }
    }

    /// Calculate bid depth (total bid volume)
    pub fn bid_depth(&self) -> f64 {
        self.bids.iter().map(|l| l.size).sum()
    }

    /// Calculate ask depth (total ask volume)
    pub fn ask_depth(&self) -> f64 {
        self.asks.iter().map(|l| l.size).sum()
    }

    /// Calculate imbalance (bid - ask) / (bid + ask)
    pub fn imbalance(&self) -> f64 {
        let bid_depth = self.bid_depth();
        let ask_depth = self.ask_depth();
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
    /// Trading symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best bid size
    pub bid_size: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Best ask size
    pub ask_size: f64,
    /// 24h price change
    pub price_change_24h: f64,
    /// 24h price change percentage
    pub price_change_percent_24h: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// Timestamp
    pub timestamp: i64,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: T,
    pub time: i64,
}

impl<T> BybitResponse<T> {
    /// Check if response is successful
    pub fn is_ok(&self) -> bool {
        self.ret_code == 0
    }
}

/// Kline result from Bybit API
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<KlineData>,
}

/// Raw kline data from Bybit (array format)
#[derive(Debug, Deserialize)]
pub struct KlineData(
    pub String, // startTime
    pub String, // openPrice
    pub String, // highPrice
    pub String, // lowPrice
    pub String, // closePrice
    pub String, // volume
    pub String, // turnover
);

impl KlineData {
    pub fn timestamp(&self) -> i64 {
        self.0.parse().unwrap_or(0)
    }
    pub fn open(&self) -> f64 {
        self.1.parse().unwrap_or(0.0)
    }
    pub fn high(&self) -> f64 {
        self.2.parse().unwrap_or(0.0)
    }
    pub fn low(&self) -> f64 {
        self.3.parse().unwrap_or(0.0)
    }
    pub fn close(&self) -> f64 {
        self.4.parse().unwrap_or(0.0)
    }
    pub fn volume(&self) -> f64 {
        self.5.parse().unwrap_or(0.0)
    }
    pub fn turnover(&self) -> f64 {
        self.6.parse().unwrap_or(0.0)
    }
}

/// Ticker result from Bybit API
#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub category: String,
    pub list: Vec<TickerData>,
}

/// Ticker data from Bybit API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerData {
    pub symbol: String,
    pub last_price: String,
    pub bid1_price: String,
    pub bid1_size: String,
    pub ask1_price: String,
    pub ask1_size: String,
    pub price_24h_pcnt: String,
    pub high_price_24h: String,
    pub low_price_24h: String,
    pub volume_24h: String,
    pub turnover_24h: String,
}

/// Order book result from Bybit API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookResult {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
    #[serde(rename = "ts")]
    pub timestamp: i64,
}
