//! Market data types
//!
//! Core data structures for representing market data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors related to market data
#[derive(Error, Debug)]
pub enum MarketDataError {
    #[error("Failed to fetch data: {0}")]
    FetchError(String),

    #[error("Invalid timeframe: {0}")]
    InvalidTimeframe(String),

    #[error("No data available for symbol: {0}")]
    NoData(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Supported timeframes for OHLCV data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFrame {
    #[serde(rename = "1m")]
    Minute1,
    #[serde(rename = "3m")]
    Minute3,
    #[serde(rename = "5m")]
    Minute5,
    #[serde(rename = "15m")]
    Minute15,
    #[serde(rename = "30m")]
    Minute30,
    #[serde(rename = "1h")]
    Hour1,
    #[serde(rename = "2h")]
    Hour2,
    #[serde(rename = "4h")]
    Hour4,
    #[serde(rename = "6h")]
    Hour6,
    #[serde(rename = "12h")]
    Hour12,
    #[serde(rename = "1d")]
    Day1,
    #[serde(rename = "1w")]
    Week1,
    #[serde(rename = "1M")]
    Month1,
}

impl TimeFrame {
    /// Convert to Bybit API interval string
    pub fn to_bybit_interval(&self) -> &'static str {
        match self {
            TimeFrame::Minute1 => "1",
            TimeFrame::Minute3 => "3",
            TimeFrame::Minute5 => "5",
            TimeFrame::Minute15 => "15",
            TimeFrame::Minute30 => "30",
            TimeFrame::Hour1 => "60",
            TimeFrame::Hour2 => "120",
            TimeFrame::Hour4 => "240",
            TimeFrame::Hour6 => "360",
            TimeFrame::Hour12 => "720",
            TimeFrame::Day1 => "D",
            TimeFrame::Week1 => "W",
            TimeFrame::Month1 => "M",
        }
    }

    /// Get the number of minutes in this timeframe
    pub fn minutes(&self) -> u32 {
        match self {
            TimeFrame::Minute1 => 1,
            TimeFrame::Minute3 => 3,
            TimeFrame::Minute5 => 5,
            TimeFrame::Minute15 => 15,
            TimeFrame::Minute30 => 30,
            TimeFrame::Hour1 => 60,
            TimeFrame::Hour2 => 120,
            TimeFrame::Hour4 => 240,
            TimeFrame::Hour6 => 360,
            TimeFrame::Hour12 => 720,
            TimeFrame::Day1 => 1440,
            TimeFrame::Week1 => 10080,
            TimeFrame::Month1 => 43200,
        }
    }
}

impl std::fmt::Display for TimeFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeFrame::Minute1 => write!(f, "1m"),
            TimeFrame::Minute3 => write!(f, "3m"),
            TimeFrame::Minute5 => write!(f, "5m"),
            TimeFrame::Minute15 => write!(f, "15m"),
            TimeFrame::Minute30 => write!(f, "30m"),
            TimeFrame::Hour1 => write!(f, "1h"),
            TimeFrame::Hour2 => write!(f, "2h"),
            TimeFrame::Hour4 => write!(f, "4h"),
            TimeFrame::Hour6 => write!(f, "6h"),
            TimeFrame::Hour12 => write!(f, "12h"),
            TimeFrame::Day1 => write!(f, "1d"),
            TimeFrame::Week1 => write!(f, "1w"),
            TimeFrame::Month1 => write!(f, "1M"),
        }
    }
}

/// OHLCV (Open, High, Low, Close, Volume) bar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    /// Timestamp of the bar
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// Highest price during the period
    pub high: f64,
    /// Lowest price during the period
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (volume * price)
    #[serde(default)]
    pub turnover: f64,
}

impl OhlcvBar {
    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate VWAP approximation for the bar
    pub fn vwap(&self) -> f64 {
        if self.volume > 0.0 && self.turnover > 0.0 {
            self.turnover / self.volume
        } else {
            self.typical_price()
        }
    }

    /// Calculate bar range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate bar body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if the bar is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Market data for a single asset
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Asset symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// OHLCV bars
    pub bars: Vec<OhlcvBar>,
    /// Timeframe of the data
    pub timeframe: TimeFrame,
}

impl MarketData {
    /// Create new market data
    pub fn new(symbol: String, timeframe: TimeFrame) -> Self {
        Self {
            symbol,
            bars: Vec::new(),
            timeframe,
        }
    }

    /// Get closing prices as a vector
    pub fn closes(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.close).collect()
    }

    /// Get volumes as a vector
    pub fn volumes(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.volume).collect()
    }

    /// Get high prices as a vector
    pub fn highs(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.high).collect()
    }

    /// Get low prices as a vector
    pub fn lows(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.low).collect()
    }

    /// Get open prices as a vector
    pub fn opens(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.open).collect()
    }

    /// Calculate simple returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return vec![];
        }

        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return vec![];
        }

        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Get the latest bar
    pub fn latest(&self) -> Option<&OhlcvBar> {
        self.bars.last()
    }

    /// Get the number of bars
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Check if there are no bars
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }
}

/// Price data for factor evaluation
#[derive(Debug, Clone, Default)]
pub struct PriceData {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub vwap: Vec<f64>,
}

impl PriceData {
    /// Create from market data
    pub fn from_market_data(data: &MarketData) -> Self {
        Self {
            open: data.opens(),
            high: data.highs(),
            low: data.lows(),
            close: data.closes(),
            volume: data.volumes(),
            vwap: data.bars.iter().map(|b| b.vwap()).collect(),
        }
    }

    /// Get the variable by name
    pub fn get(&self, name: &str) -> Option<&Vec<f64>> {
        match name.to_lowercase().as_str() {
            "open" => Some(&self.open),
            "high" => Some(&self.high),
            "low" => Some(&self.low),
            "close" => Some(&self.close),
            "volume" | "vol" => Some(&self.volume),
            "vwap" => Some(&self.vwap),
            _ => None,
        }
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

/// Current ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub price_change_24h: f64,
    pub timestamp: DateTime<Utc>,
}

impl Ticker {
    /// Calculate spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Calculate spread percentage
    pub fn spread_percent(&self) -> f64 {
        if self.bid_price > 0.0 {
            (self.ask_price - self.bid_price) / self.bid_price * 100.0
        } else {
            0.0
        }
    }
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>, // (price, quantity)
    pub timestamp: DateTime<Utc>,
}

impl OrderBook {
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate book imbalance
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_qty: f64 = self.bids.iter().take(levels).map(|(_, q)| q).sum();
        let ask_qty: f64 = self.asks.iter().take(levels).map(|(_, q)| q).sum();

        if bid_qty + ask_qty > 0.0 {
            (bid_qty - ask_qty) / (bid_qty + ask_qty)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_calculations() {
        let bar = OhlcvBar {
            timestamp: Utc::now(),
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 102500.0,
        };

        assert_eq!(bar.typical_price(), (110.0 + 95.0 + 105.0) / 3.0);
        assert_eq!(bar.range(), 15.0);
        assert_eq!(bar.body(), 5.0);
        assert!(bar.is_bullish());
        assert!((bar.vwap() - 102.5).abs() < 0.001);
    }

    #[test]
    fn test_market_data_returns() {
        let mut data = MarketData::new("BTCUSDT".to_string(), TimeFrame::Day1);
        data.bars.push(OhlcvBar {
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 100.0,
            volume: 1000.0,
            turnover: 100000.0,
        });
        data.bars.push(OhlcvBar {
            timestamp: Utc::now(),
            open: 100.0,
            high: 110.0,
            low: 99.0,
            close: 110.0,
            volume: 1500.0,
            turnover: 160000.0,
        });

        let returns = data.returns();
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_order_book_imbalance() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![(100.0, 10.0), (99.0, 5.0)],
            asks: vec![(101.0, 5.0), (102.0, 5.0)],
            timestamp: Utc::now(),
        };

        let imbalance = book.imbalance(2);
        // bid_qty = 15, ask_qty = 10, imbalance = 5/25 = 0.2
        assert!((imbalance - 0.2).abs() < 0.001);
    }
}
