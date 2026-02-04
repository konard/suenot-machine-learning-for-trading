//! API data types for Bybit exchange.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp of the candle
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
    #[serde(default)]
    pub turnover: f64,
}

impl Kline {
    /// Create a new Kline from raw values.
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
            turnover: 0.0,
        }
    }

    /// Calculate the return from open to close.
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate the range (high - low) as percentage of close.
    pub fn range_pct(&self) -> f64 {
        (self.high - self.low) / self.close
    }

    /// Check if this is a bullish candle.
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate the body size (absolute).
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate upper shadow size.
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate lower shadow size.
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }
}

/// Ticker data for a trading pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Trading pair symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high price
    pub high_24h: f64,
    /// 24h low price
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// Price change percentage
    pub price_change_pct: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Ticker {
    /// Calculate bid-ask spread in basis points.
    pub fn spread_bps(&self) -> f64 {
        let mid = (self.bid_price + self.ask_price) / 2.0;
        (self.ask_price - self.bid_price) / mid * 10000.0
    }
}

/// Order book data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading pair symbol
    pub symbol: String,
    /// Bid levels (price, quantity)
    pub bids: Vec<(f64, f64)>,
    /// Ask levels (price, quantity)
    pub asks: Vec<(f64, f64)>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
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

    /// Calculate mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate spread in basis points.
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask(), self.mid_price()) {
            (Some(bid), Some(ask), Some(mid)) => Some((ask - bid) / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate bid depth (total bid volume).
    pub fn bid_depth(&self) -> f64 {
        self.bids.iter().map(|(_, q)| q).sum()
    }

    /// Calculate ask depth (total ask volume).
    pub fn ask_depth(&self) -> f64 {
        self.asks.iter().map(|(_, q)| q).sum()
    }

    /// Calculate order book imbalance.
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

/// Individual trade data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Trading pair symbol
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Trade side (buy/sell)
    pub side: TradeSide,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Trade side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// API response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: T,
    pub time: u64,
}

/// Klines response from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}
