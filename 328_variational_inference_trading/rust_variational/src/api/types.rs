//! API types for Bybit exchange
//!
//! Defines data structures for market data from Bybit.

use serde::{Deserialize, Serialize};

/// API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Ticker list result
#[derive(Debug, Deserialize)]
pub struct TickerListResult {
    pub list: Vec<Ticker>,
}

/// Market ticker data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Ticker {
    pub symbol: String,
    #[serde(rename = "lastPrice", deserialize_with = "parse_f64")]
    pub last_price: f64,
    #[serde(rename = "highPrice24h", deserialize_with = "parse_f64")]
    pub high_price_24h: f64,
    #[serde(rename = "lowPrice24h", deserialize_with = "parse_f64")]
    pub low_price_24h: f64,
    #[serde(rename = "volume24h", deserialize_with = "parse_f64")]
    pub volume_24h: f64,
    #[serde(rename = "turnover24h", deserialize_with = "parse_f64")]
    pub turnover_24h: f64,
    #[serde(rename = "price24hPcnt", deserialize_with = "parse_f64")]
    pub price_change_24h: f64,
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub start_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl Kline {
    /// Parse kline from API response array
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

    /// Calculate typical price
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body (absolute difference between open and close)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Kline list result
#[derive(Debug, Deserialize)]
pub struct KlineListResult {
    pub list: Vec<Vec<String>>,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: f64,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: i64,
    pub update_id: i64,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
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
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid + ask) / 2.0;
                Some((ask - bid) / mid * 10000.0)
            }
            _ => None,
        }
    }

    /// Calculate depth imbalance at given levels
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid_depth: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();
        let ask_depth: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();

        if bid_depth + ask_depth > 0.0 {
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
        } else {
            0.0
        }
    }
}

/// Order book API result
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String,
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
    pub ts: i64,
    pub u: i64,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub is_buyer_maker: bool,
    pub timestamp: i64,
}

impl Trade {
    /// Check if this is a buy trade
    pub fn is_buy(&self) -> bool {
        !self.is_buyer_maker
    }

    /// Calculate trade value
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// Helper function to parse f64 from string
fn parse_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_from_response() {
        let data = vec![
            "1704067200000".to_string(),
            "42000.50".to_string(),
            "42500.00".to_string(),
            "41800.00".to_string(),
            "42200.00".to_string(),
            "1000.5".to_string(),
            "42100000.0".to_string(),
        ];

        let kline = Kline::from_api_response(&data).unwrap();
        assert_eq!(kline.start_time, 1704067200000);
        assert!((kline.open - 42000.50).abs() < 0.01);
        assert!(kline.is_bullish());
    }

    #[test]
    fn test_order_book() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 41999.0, size: 1.0 },
                OrderBookLevel { price: 41998.0, size: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 42001.0, size: 1.0 },
                OrderBookLevel { price: 42002.0, size: 2.0 },
            ],
            timestamp: 0,
            update_id: 0,
        };

        assert_eq!(book.best_bid(), Some(41999.0));
        assert_eq!(book.best_ask(), Some(42001.0));
        assert!((book.mid_price().unwrap() - 42000.0).abs() < 0.01);
    }
}
