//! Market data structures and utilities

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Candle open time
    pub timestamp: DateTime<Utc>,
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
}

impl OHLCV {
    /// Create a new OHLCV candle
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
        }
    }

    /// Check if candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Get candle body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get candle range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get upper wick size
    pub fn upper_wick(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Get lower wick size
    pub fn lower_wick(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Get midpoint price
    pub fn midpoint(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Get typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Real-time ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last trade price
    pub last_price: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h price change percentage
    pub change_24h: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Ticker {
    /// Get bid-ask spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Get spread as percentage of mid price
    pub fn spread_pct(&self) -> f64 {
        let mid = (self.bid_price + self.ask_price) / 2.0;
        if mid == 0.0 {
            return 0.0;
        }
        (self.spread() / mid) * 100.0
    }

    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
}

/// Market data provider trait
#[async_trait::async_trait]
pub trait MarketData: Send + Sync {
    /// Get current ticker for a symbol
    async fn get_ticker(&self, symbol: &str) -> Result<Ticker, MarketDataError>;

    /// Get historical OHLCV data
    async fn get_ohlcv(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<OHLCV>, MarketDataError>;

    /// Get order book depth
    async fn get_orderbook(&self, symbol: &str, depth: usize) -> Result<OrderBook, MarketDataError>;

    /// Check if connected
    fn is_connected(&self) -> bool;
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid orders (price, quantity)
    pub bids: Vec<(f64, f64)>,
    /// Ask orders (price, quantity)
    pub asks: Vec<(f64, f64)>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(price, _)| *price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(price, _)| *price)
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get total bid volume up to a price
    pub fn bid_volume_to_price(&self, price: f64) -> f64 {
        self.bids
            .iter()
            .filter(|(p, _)| *p >= price)
            .map(|(_, q)| q)
            .sum()
    }

    /// Get total ask volume up to a price
    pub fn ask_volume_to_price(&self, price: f64) -> f64 {
        self.asks
            .iter()
            .filter(|(p, _)| *p <= price)
            .map(|(_, q)| q)
            .sum()
    }

    /// Calculate imbalance ratio (bid volume / total volume)
    pub fn imbalance(&self) -> f64 {
        let bid_vol: f64 = self.bids.iter().map(|(_, q)| q).sum();
        let ask_vol: f64 = self.asks.iter().map(|(_, q)| q).sum();
        let total = bid_vol + ask_vol;

        if total == 0.0 {
            0.5
        } else {
            bid_vol / total
        }
    }
}

/// Market data errors
#[derive(Debug, thiserror::Error)]
pub enum MarketDataError {
    /// Symbol not found
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),

    /// Connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {0} seconds")]
    RateLimitError(u64),

    /// API error
    #[error("API error: {0}")]
    ApiError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_bullish_bearish() {
        let bullish = OHLCV::new(Utc::now(), 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());

        let bearish = OHLCV::new(Utc::now(), 100.0, 105.0, 90.0, 95.0, 1000.0);
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
    }

    #[test]
    fn test_ohlcv_calculations() {
        let candle = OHLCV::new(Utc::now(), 100.0, 110.0, 90.0, 105.0, 1000.0);

        assert_eq!(candle.body_size(), 5.0);
        assert_eq!(candle.range(), 20.0);
        assert_eq!(candle.upper_wick(), 5.0);
        assert_eq!(candle.lower_wick(), 10.0);
        assert_eq!(candle.midpoint(), 100.0);
    }

    #[test]
    fn test_ticker_spread() {
        let ticker = Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            bid_price: 49990.0,
            ask_price: 50010.0,
            volume_24h: 1000000.0,
            change_24h: 2.5,
            timestamp: Utc::now(),
        };

        assert_eq!(ticker.spread(), 20.0);
        assert_eq!(ticker.mid_price(), 50000.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![(50000.0, 10.0), (49990.0, 5.0)],
            asks: vec![(50010.0, 5.0), (50020.0, 5.0)],
            timestamp: Utc::now(),
        };

        // Bid volume = 15, Ask volume = 10, Total = 25
        // Imbalance = 15/25 = 0.6
        assert!((book.imbalance() - 0.6).abs() < 0.001);
    }
}
