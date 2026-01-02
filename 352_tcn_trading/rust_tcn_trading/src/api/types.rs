//! API Types and Data Structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Candlestick/OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp
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
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Calculate the return for this candle
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get the body size
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get the upper shadow
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Get the lower shadow
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// Market data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Trading symbol
    pub symbol: String,
    /// Candle data
    pub candles: Vec<Candle>,
    /// Timeframe
    pub timeframe: Option<TimeFrame>,
}

impl MarketData {
    /// Create new market data
    pub fn new(symbol: &str, candles: Vec<Candle>) -> Self {
        Self {
            symbol: symbol.to_string(),
            candles,
            timeframe: None,
        }
    }

    /// Get the number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get closing prices
    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get opening prices
    pub fn opens(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.open).collect()
    }

    /// Get high prices
    pub fn highs(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.high).collect()
    }

    /// Get low prices
    pub fn lows(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.low).collect()
    }

    /// Get volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Calculate returns
    pub fn returns(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.return_pct()).collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }
}

/// Timeframe enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFrame {
    /// 1 minute
    Min1,
    /// 3 minutes
    Min3,
    /// 5 minutes
    Min5,
    /// 15 minutes
    Min15,
    /// 30 minutes
    Min30,
    /// 1 hour
    Hour1,
    /// 2 hours
    Hour2,
    /// 4 hours
    Hour4,
    /// 6 hours
    Hour6,
    /// 12 hours
    Hour12,
    /// 1 day
    Day1,
    /// 1 week
    Week1,
    /// 1 month
    Month1,
}

impl TimeFrame {
    /// Convert to Bybit API string
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeFrame::Min1 => "1",
            TimeFrame::Min3 => "3",
            TimeFrame::Min5 => "5",
            TimeFrame::Min15 => "15",
            TimeFrame::Min30 => "30",
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

    /// Get duration in minutes
    pub fn minutes(&self) -> u64 {
        match self {
            TimeFrame::Min1 => 1,
            TimeFrame::Min3 => 3,
            TimeFrame::Min5 => 5,
            TimeFrame::Min15 => 15,
            TimeFrame::Min30 => 30,
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

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1m" | "1" => Some(TimeFrame::Min1),
            "3m" | "3" => Some(TimeFrame::Min3),
            "5m" | "5" => Some(TimeFrame::Min5),
            "15m" | "15" => Some(TimeFrame::Min15),
            "30m" | "30" => Some(TimeFrame::Min30),
            "1h" | "60" => Some(TimeFrame::Hour1),
            "2h" | "120" => Some(TimeFrame::Hour2),
            "4h" | "240" => Some(TimeFrame::Hour4),
            "6h" | "360" => Some(TimeFrame::Hour6),
            "12h" | "720" => Some(TimeFrame::Hour12),
            "1d" | "d" => Some(TimeFrame::Day1),
            "1w" | "w" => Some(TimeFrame::Week1),
            "1M" | "m" => Some(TimeFrame::Month1),
            _ => None,
        }
    }
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Bid levels (price, quantity)
    pub bids: Vec<(f64, f64)>,
    /// Ask levels (price, quantity)
    pub asks: Vec<(f64, f64)>,
}

impl OrderBookSnapshot {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate bid-ask imbalance
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_vol: f64 = self.bids.iter().take(levels).map(|(_, q)| q).sum();
        let ask_vol: f64 = self.asks.iter().take(levels).map(|(_, q)| q).sum();
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

/// Ticker information
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
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h price change
    pub price_change_24h: f64,
    /// 24h price change percentage
    pub price_change_pct_24h: f64,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Kline result from API
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Order book result from API
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String,
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
    pub ts: i64,
    pub u: i64,
}

/// Ticker result from API
#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub category: String,
    pub list: Vec<TickerItem>,
}

/// Individual ticker item
#[derive(Debug, Deserialize)]
pub struct TickerItem {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "bid1Price")]
    pub bid1_price: String,
    #[serde(rename = "ask1Price")]
    pub ask1_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_calculations() {
        let candle = Candle::new(
            Utc::now(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
            100000.0,
        );

        assert_eq!(candle.range(), 15.0);
        assert!(candle.is_bullish());
        assert_eq!(candle.body(), 5.0);
        assert_eq!(candle.upper_shadow(), 5.0);
        assert_eq!(candle.lower_shadow(), 5.0);
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(TimeFrame::Hour1.as_str(), "60");
        assert_eq!(TimeFrame::Day1.as_str(), "D");
        assert_eq!(TimeFrame::Hour1.minutes(), 60);
    }

    #[test]
    fn test_orderbook_calculations() {
        let ob = OrderBookSnapshot {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            bids: vec![(100.0, 10.0), (99.0, 20.0)],
            asks: vec![(101.0, 15.0), (102.0, 25.0)],
        };

        assert_eq!(ob.best_bid(), Some(100.0));
        assert_eq!(ob.best_ask(), Some(101.0));
        assert_eq!(ob.mid_price(), Some(100.5));
        assert_eq!(ob.spread(), Some(1.0));

        // Imbalance with 2 levels
        let imb = ob.imbalance(2);
        // bid_vol = 30, ask_vol = 40, imbalance = (30-40)/70 = -0.142...
        assert!((imb - (-10.0 / 70.0)).abs() < 0.001);
    }
}
