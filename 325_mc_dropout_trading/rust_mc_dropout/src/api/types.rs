//! API types for Bybit data structures

use serde::{Deserialize, Serialize};

/// Ticker information for a trading pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub timestamp: i64,
}

impl Ticker {
    /// Calculate the spread in basis points
    pub fn spread_bps(&self) -> f64 {
        let mid = (self.bid_price + self.ask_price) / 2.0;
        if mid > 0.0 {
            (self.ask_price - self.bid_price) / mid * 10000.0
        } else {
            0.0
        }
    }

    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub start_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Kline {
    /// Calculate return from open to close
    pub fn return_oc(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate range as percentage of close
    pub fn range_pct(&self) -> f64 {
        if self.close > 0.0 {
            self.range() / self.close
        } else {
            0.0
        }
    }

    /// Check if candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: i64,
}

impl OrderBook {
    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            Some((best_bid.price + best_ask.price) / 2.0)
        } else {
            None
        }
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            let mid = (best_bid.price + best_ask.price) / 2.0;
            if mid > 0.0 {
                Some((best_ask.price - best_bid.price) / mid * 10000.0)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Calculate depth imbalance
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();
        let ask_volume: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();
        let total = bid_volume + ask_volume;

        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }
}

/// Individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: i64,
    pub price: f64,
    pub size: f64,
    pub side: String,
}

impl Trade {
    /// Check if this is a buy trade
    pub fn is_buy(&self) -> bool {
        self.side.to_lowercase() == "buy"
    }

    /// Get trade value (price * size)
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// API response wrapper for Bybit v5 API
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Ticker response from Bybit API
#[derive(Debug, Deserialize)]
pub struct TickerResponse {
    pub list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
pub struct TickerItem {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "bid1Price")]
    pub bid_price: String,
    #[serde(rename = "ask1Price")]
    pub ask_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_change_24h: String,
    #[serde(rename = "highPrice24h")]
    pub high_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_24h: String,
}

/// Kline response from Bybit API
#[derive(Debug, Deserialize)]
pub struct KlineResponse {
    pub list: Vec<Vec<String>>,
}

/// Order book response from Bybit API
#[derive(Debug, Deserialize)]
pub struct OrderBookResponse {
    pub s: String, // symbol
    pub b: Vec<Vec<String>>, // bids
    pub a: Vec<Vec<String>>, // asks
    pub ts: i64, // timestamp
}
