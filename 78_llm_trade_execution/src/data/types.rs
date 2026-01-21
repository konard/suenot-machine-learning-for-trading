//! Common market data types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Market data errors
#[derive(Error, Debug)]
pub enum MarketDataError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Exchange error: {code} - {message}")]
    Exchange { code: i32, message: String },

    #[error("Data not available: {0}")]
    NotAvailable(String),
}

/// Time frame for OHLCV data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeFrame {
    /// 1 minute
    M1,
    /// 3 minutes
    M3,
    /// 5 minutes
    M5,
    /// 15 minutes
    M15,
    /// 30 minutes
    M30,
    /// 1 hour
    H1,
    /// 2 hours
    H2,
    /// 4 hours
    H4,
    /// 6 hours
    H6,
    /// 12 hours
    H12,
    /// 1 day
    D1,
    /// 1 week
    W1,
    /// 1 month
    MN,
}

impl TimeFrame {
    /// Get the duration in seconds
    pub fn as_seconds(&self) -> u64 {
        match self {
            TimeFrame::M1 => 60,
            TimeFrame::M3 => 180,
            TimeFrame::M5 => 300,
            TimeFrame::M15 => 900,
            TimeFrame::M30 => 1800,
            TimeFrame::H1 => 3600,
            TimeFrame::H2 => 7200,
            TimeFrame::H4 => 14400,
            TimeFrame::H6 => 21600,
            TimeFrame::H12 => 43200,
            TimeFrame::D1 => 86400,
            TimeFrame::W1 => 604800,
            TimeFrame::MN => 2592000, // 30 days approximation
        }
    }

    /// Convert to Bybit interval string
    pub fn to_bybit_interval(&self) -> &'static str {
        match self {
            TimeFrame::M1 => "1",
            TimeFrame::M3 => "3",
            TimeFrame::M5 => "5",
            TimeFrame::M15 => "15",
            TimeFrame::M30 => "30",
            TimeFrame::H1 => "60",
            TimeFrame::H2 => "120",
            TimeFrame::H4 => "240",
            TimeFrame::H6 => "360",
            TimeFrame::H12 => "720",
            TimeFrame::D1 => "D",
            TimeFrame::W1 => "W",
            TimeFrame::MN => "M",
        }
    }
}

impl fmt::Display for TimeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeFrame::M1 => write!(f, "1m"),
            TimeFrame::M3 => write!(f, "3m"),
            TimeFrame::M5 => write!(f, "5m"),
            TimeFrame::M15 => write!(f, "15m"),
            TimeFrame::M30 => write!(f, "30m"),
            TimeFrame::H1 => write!(f, "1h"),
            TimeFrame::H2 => write!(f, "2h"),
            TimeFrame::H4 => write!(f, "4h"),
            TimeFrame::H6 => write!(f, "6h"),
            TimeFrame::H12 => write!(f, "12h"),
            TimeFrame::D1 => write!(f, "1d"),
            TimeFrame::W1 => write!(f, "1w"),
            TimeFrame::MN => write!(f, "1M"),
        }
    }
}

/// OHLCV bar (candlestick)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    /// Bar timestamp (start time)
    pub timestamp: DateTime<Utc>,
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
    /// Turnover (quote currency volume)
    pub turnover: Option<f64>,
}

impl OhlcvBar {
    /// Create a new OHLCV bar
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: None,
        }
    }

    /// Get the typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get the VWAP approximation for this bar
    pub fn vwap(&self) -> f64 {
        self.typical_price()
    }

    /// Check if the bar is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get the bar range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    /// Buy (taker was buyer)
    Buy,
    /// Sell (taker was seller)
    Sell,
}

/// Individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Trade direction (buy/sell)
    pub direction: TradeDirection,
}

impl Trade {
    /// Get the trade value (price * quantity)
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Volume information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Volume {
    /// Total volume
    pub total: f64,
    /// Buy volume
    pub buy: f64,
    /// Sell volume
    pub sell: f64,
    /// Number of trades
    pub trade_count: u64,
}

impl Volume {
    /// Get the buy/sell ratio
    pub fn buy_sell_ratio(&self) -> f64 {
        if self.sell > 0.0 {
            self.buy / self.sell
        } else {
            f64::INFINITY
        }
    }

    /// Get the net volume (buy - sell)
    pub fn net(&self) -> f64 {
        self.buy - self.sell
    }
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Best bid quantity
    pub bid_qty: f64,
    /// Best ask quantity
    pub ask_qty: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// Open interest (for derivatives)
    pub open_interest: Option<f64>,
    /// Funding rate (for perpetuals)
    pub funding_rate: Option<f64>,
    /// Next funding time (for perpetuals)
    pub next_funding_time: Option<DateTime<Utc>>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Ticker {
    /// Get the mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    /// Get the spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Get the spread in basis points
    pub fn spread_bps(&self) -> f64 {
        (self.spread() / self.mid_price()) * 10000.0
    }
}

/// Price data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Price
    pub price: f64,
    /// Volume at this price level
    pub volume: Option<f64>,
}

/// Market data container
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    /// Symbol
    pub symbol: String,
    /// Current ticker
    pub ticker: Option<Ticker>,
    /// Recent OHLCV bars
    pub bars: Vec<OhlcvBar>,
    /// Recent trades
    pub trades: Vec<Trade>,
    /// Volume profile
    pub volume: Volume,
}

impl MarketData {
    /// Create new market data for a symbol
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            ..Default::default()
        }
    }

    /// Get the current price (last trade or mid price)
    pub fn current_price(&self) -> Option<f64> {
        self.ticker
            .as_ref()
            .map(|t| t.last_price)
            .or_else(|| self.trades.last().map(|t| t.price))
    }

    /// Get the current spread
    pub fn current_spread(&self) -> Option<f64> {
        self.ticker.as_ref().map(|t| t.spread())
    }

    /// Calculate VWAP from recent trades
    pub fn calculate_vwap(&self, num_trades: usize) -> Option<f64> {
        let trades: Vec<_> = self.trades.iter().rev().take(num_trades).collect();
        if trades.is_empty() {
            return None;
        }

        let total_value: f64 = trades.iter().map(|t| t.value()).sum();
        let total_volume: f64 = trades.iter().map(|t| t.quantity).sum();

        if total_volume > 0.0 {
            Some(total_value / total_volume)
        } else {
            None
        }
    }

    /// Get average trade size
    pub fn average_trade_size(&self, num_trades: usize) -> Option<f64> {
        let trades: Vec<_> = self.trades.iter().rev().take(num_trades).collect();
        if trades.is_empty() {
            return None;
        }

        let total_qty: f64 = trades.iter().map(|t| t.quantity).sum();
        Some(total_qty / trades.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_seconds() {
        assert_eq!(TimeFrame::M1.as_seconds(), 60);
        assert_eq!(TimeFrame::H1.as_seconds(), 3600);
        assert_eq!(TimeFrame::D1.as_seconds(), 86400);
    }

    #[test]
    fn test_ohlcv_bar() {
        let bar = OhlcvBar::new(Utc::now(), 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert!(bar.is_bullish());
        assert_eq!(bar.range(), 15.0);
        assert_eq!(bar.body_size(), 5.0);
        assert!((bar.typical_price() - 103.333).abs() < 0.01);
    }

    #[test]
    fn test_volume_ratio() {
        let volume = Volume {
            total: 1000.0,
            buy: 600.0,
            sell: 400.0,
            trade_count: 100,
        };
        assert!((volume.buy_sell_ratio() - 1.5).abs() < 0.001);
        assert!((volume.net() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_ticker_spread() {
        let ticker = Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            bid_price: 49990.0,
            ask_price: 50010.0,
            bid_qty: 1.0,
            ask_qty: 1.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            volume_24h: 10000.0,
            turnover_24h: 500000000.0,
            open_interest: Some(100000.0),
            funding_rate: Some(0.0001),
            next_funding_time: None,
            timestamp: Utc::now(),
        };

        assert_eq!(ticker.mid_price(), 50000.0);
        assert_eq!(ticker.spread(), 20.0);
        assert!((ticker.spread_bps() - 4.0).abs() < 0.001);
    }
}
