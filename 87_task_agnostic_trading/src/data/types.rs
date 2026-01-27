//! Data types for market information

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time of the candle
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
    pub turnover: f64,
}

impl Kline {
    /// Calculate the return for this candle
    pub fn returns(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate the range (high-low) as percentage
    pub fn range_pct(&self) -> f64 {
        if self.low > 0.0 {
            (self.high - self.low) / self.low
        } else {
            0.0
        }
    }

    /// Calculate body size (abs(close-open)) as percentage
    pub fn body_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open).abs() / self.open
        } else {
            0.0
        }
    }

    /// Whether the candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Whether the trade was a buy
    pub is_buy: bool,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Bid levels (price, quantity)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (price, quantity)
    pub asks: Vec<OrderBookLevel>,
}

/// Single order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub quantity: f64,
}

impl OrderBook {
    /// Calculate bid-ask spread
    pub fn spread(&self) -> f64 {
        let best_bid = self.bids.first().map(|l| l.price).unwrap_or(0.0);
        let best_ask = self.asks.first().map(|l| l.price).unwrap_or(0.0);
        if best_bid > 0.0 {
            (best_ask - best_bid) / best_bid
        } else {
            0.0
        }
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self) -> f64 {
        let bid_vol: f64 = self.bids.iter().map(|l| l.quantity).sum();
        let ask_vol: f64 = self.asks.iter().map(|l| l.quantity).sum();
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

/// Funding rate data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Funding rate value
    pub rate: f64,
}

/// Ticker data (24h summary)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
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
    /// 24h price change percentage
    pub price_change_pct: f64,
}
