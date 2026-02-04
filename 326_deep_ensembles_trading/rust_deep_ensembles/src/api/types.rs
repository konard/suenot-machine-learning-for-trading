//! API type definitions for Bybit responses

use serde::{Deserialize, Serialize};

/// Generic API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

impl Ticker {
    pub fn last_price_f64(&self) -> Option<f64> {
        self.last_price.parse().ok()
    }

    pub fn volume_f64(&self) -> Option<f64> {
        self.volume_24h.parse().ok()
    }
}

/// Ticker list result
#[derive(Debug, Deserialize)]
pub struct TickerListResult {
    pub list: Vec<Ticker>,
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
    /// Create Kline from API response array
    pub fn from_api_response(data: &[String]) -> Option<Self> {
        if data.len() < 7 {
            return None;
        }

        Some(Kline {
            start_time: data[0].parse().ok()?,
            open: data[1].parse().ok()?,
            high: data[2].parse().ok()?,
            low: data[3].parse().ok()?,
            close: data[4].parse().ok()?,
            volume: data[5].parse().ok()?,
            turnover: data[6].parse().ok()?,
        })
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate candle range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body size
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
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.price;
        let best_ask = self.asks.first()?.price;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.price;
        let best_ask = self.asks.first()?.price;
        let mid = (best_bid + best_ask) / 2.0;
        Some((best_ask - best_bid) / mid * 10000.0)
    }

    /// Calculate bid-ask imbalance
    pub fn imbalance(&self) -> f64 {
        let bid_volume: f64 = self.bids.iter().map(|l| l.size).sum();
        let ask_volume: f64 = self.asks.iter().map(|l| l.size).sum();
        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
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
        assert!(kline.is_bullish());
    }

    #[test]
    fn test_orderbook_spread() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![OrderBookLevel { price: 99.95, size: 10.0 }],
            asks: vec![OrderBookLevel { price: 100.05, size: 10.0 }],
            timestamp: 0,
            update_id: 0,
        };

        let spread = orderbook.spread_bps().unwrap();
        assert!((spread - 10.0).abs() < 0.1); // ~10 bps
    }
}
