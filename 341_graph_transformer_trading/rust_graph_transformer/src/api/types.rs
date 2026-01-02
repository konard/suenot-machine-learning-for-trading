//! API data types for Bybit exchange
//!
//! Defines structures for market data received from Bybit API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Ticker data for a trading pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Last traded price
    #[serde(deserialize_with = "deserialize_f64_from_string")]
    pub last_price: f64,
    /// 24h high price
    #[serde(rename = "highPrice24h", deserialize_with = "deserialize_f64_from_string")]
    pub high_24h: f64,
    /// 24h low price
    #[serde(rename = "lowPrice24h", deserialize_with = "deserialize_f64_from_string")]
    pub low_24h: f64,
    /// 24h price change percentage
    #[serde(rename = "price24hPcnt", deserialize_with = "deserialize_f64_from_string")]
    pub price_change_24h: f64,
    /// 24h volume in base currency
    #[serde(rename = "volume24h", deserialize_with = "deserialize_f64_from_string")]
    pub volume_24h: f64,
    /// 24h turnover in quote currency
    #[serde(rename = "turnover24h", deserialize_with = "deserialize_f64_from_string")]
    pub turnover_24h: f64,
    /// Best bid price
    #[serde(rename = "bid1Price", deserialize_with = "deserialize_f64_from_string")]
    pub bid_price: f64,
    /// Best ask price
    #[serde(rename = "ask1Price", deserialize_with = "deserialize_f64_from_string")]
    pub ask_price: f64,
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp
    pub start_time: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume in base currency
    pub volume: f64,
    /// Turnover in quote currency
    pub turnover: f64,
}

impl Kline {
    /// Create a new Kline from raw API response
    pub fn from_api_response(data: &[String]) -> Option<Self> {
        if data.len() < 7 {
            return None;
        }
        Some(Self {
            start_time: data[0].parse().ok()?,
            open: data[1].parse().ok()?,
            high: data[2].parse().ok()?,
            low: data[3].parse().ok()?,
            close: data[4].parse().ok()?,
            volume: data[5].parse().ok()?,
            turnover: data[6].parse().ok()?,
        })
    }

    /// Calculate the return from open to close
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate the range (high - low) / open
    pub fn range_pct(&self) -> f64 {
        (self.high - self.low) / self.open
    }

    /// Get the typical price (H+L+C)/3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Order book level with price and size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Size at this price level
    pub size: f64,
}

/// Full order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: i64,
    /// Update ID
    pub update_id: u64,
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

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask(), self.mid_price()) {
            (Some(bid), Some(ask), Some(mid)) => Some((ask - bid) / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate depth imbalance at top N levels
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

    /// Calculate total bid depth
    pub fn total_bid_depth(&self) -> f64 {
        self.bids.iter().map(|l| l.size).sum()
    }

    /// Calculate total ask depth
    pub fn total_ask_depth(&self) -> f64 {
        self.asks.iter().map(|l| l.size).sum()
    }
}

/// Individual trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Trading symbol
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// Trade side (true = buy, false = sell)
    pub is_buyer_maker: bool,
    /// Trade timestamp
    pub timestamp: i64,
}

impl Trade {
    /// Check if this is a buy trade
    pub fn is_buy(&self) -> bool {
        !self.is_buyer_maker
    }

    /// Get trade value (price * size)
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
    pub time: i64,
}

/// Ticker list result
#[derive(Debug, Deserialize)]
pub struct TickerListResult {
    pub category: String,
    pub list: Vec<Ticker>,
}

/// Kline list result
#[derive(Debug, Deserialize)]
pub struct KlineListResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Order book result
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String,  // symbol
    pub b: Vec<Vec<String>>,  // bids
    pub a: Vec<Vec<String>>,  // asks
    pub ts: i64,  // timestamp
    pub u: u64,  // update id
}

/// Helper function to deserialize f64 from string
fn deserialize_f64_from_string<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    s.parse::<f64>().map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_return() {
        let kline = Kline {
            start_time: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };
        assert!((kline.return_pct() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_orderbook_spread() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 99.0, size: 10.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, size: 10.0 },
            ],
            timestamp: 0,
            update_id: 0,
        };
        let spread = ob.spread_bps().unwrap();
        assert!((spread - 200.0).abs() < 1.0);  // 2% spread = 200 bps
    }
}
