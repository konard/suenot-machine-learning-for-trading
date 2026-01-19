//! API types for Bybit exchange
//!
//! Data structures for API responses and market data.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bybit API error types
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: code={code}, message={message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Insufficient data: need {need}, got {got}")]
    InsufficientData { need: usize, got: usize },
}

/// Generic API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: Option<T>,
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Opening timestamp in milliseconds
    pub timestamp: u64,
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
    /// Turnover (volume * price)
    pub turnover: f64,
}

impl Kline {
    /// Parse kline from Bybit API array format
    pub fn from_bybit_array(arr: &[String]) -> Result<Self, BybitError> {
        if arr.len() < 7 {
            return Err(BybitError::ParseError(format!(
                "Invalid kline array length: {}",
                arr.len()
            )));
        }

        Ok(Self {
            timestamp: arr[0]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid timestamp".to_string()))?,
            open: arr[1]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid open".to_string()))?,
            high: arr[2]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid high".to_string()))?,
            low: arr[3]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid low".to_string()))?,
            close: arr[4]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid close".to_string()))?,
            volume: arr[5]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid volume".to_string()))?,
            turnover: arr[6]
                .parse()
                .map_err(|_| BybitError::ParseError("Invalid turnover".to_string()))?,
        })
    }

    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body size (absolute)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Klines API result
#[derive(Debug, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub index_price: f64,
    pub mark_price: f64,
    pub prev_price_24h: f64,
    pub price_24h_pcnt: f64,
    pub high_price_24h: f64,
    pub low_price_24h: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub open_interest: f64,
    pub funding_rate: f64,
    pub next_funding_time: u64,
}

/// Raw ticker from API
#[derive(Debug, Deserialize)]
pub struct RawTicker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "indexPrice")]
    pub index_price: String,
    #[serde(rename = "markPrice")]
    pub mark_price: String,
    #[serde(rename = "prevPrice24h")]
    pub prev_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "openInterest")]
    pub open_interest: String,
    #[serde(rename = "fundingRate")]
    pub funding_rate: String,
    #[serde(rename = "nextFundingTime")]
    pub next_funding_time: String,
}

impl RawTicker {
    pub fn to_ticker(&self) -> Result<Ticker, BybitError> {
        Ok(Ticker {
            symbol: self.symbol.clone(),
            last_price: self.last_price.parse().unwrap_or(0.0),
            index_price: self.index_price.parse().unwrap_or(0.0),
            mark_price: self.mark_price.parse().unwrap_or(0.0),
            prev_price_24h: self.prev_price_24h.parse().unwrap_or(0.0),
            price_24h_pcnt: self.price_24h_pcnt.parse().unwrap_or(0.0),
            high_price_24h: self.high_price_24h.parse().unwrap_or(0.0),
            low_price_24h: self.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: self.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: self.turnover_24h.parse().unwrap_or(0.0),
            open_interest: self.open_interest.parse().unwrap_or(0.0),
            funding_rate: self.funding_rate.parse().unwrap_or(0.0),
            next_funding_time: self.next_funding_time.parse().unwrap_or(0),
        })
    }
}

/// Tickers API result
#[derive(Debug, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<RawTicker>,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.first().map(|l| l.price)?;
        let best_ask = self.asks.first().map(|l| l.price)?;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Calculate bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        let best_bid = self.bids.first().map(|l| l.price)?;
        let best_ask = self.asks.first().map(|l| l.price)?;
        Some(best_ask - best_bid)
    }

    /// Calculate spread percentage
    pub fn spread_pct(&self) -> Option<f64> {
        let mid = self.mid_price()?;
        let spread = self.spread()?;
        Some(spread / mid * 100.0)
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_vol: f64 = self.bids.iter().take(levels).map(|l| l.quantity).sum();
        let ask_vol: f64 = self.asks.iter().take(levels).map(|l| l.quantity).sum();
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_from_array() {
        let arr = vec![
            "1704067200000".to_string(),
            "42000.0".to_string(),
            "42500.0".to_string(),
            "41800.0".to_string(),
            "42300.0".to_string(),
            "1000.5".to_string(),
            "42150000.0".to_string(),
        ];

        let kline = Kline::from_bybit_array(&arr).unwrap();

        assert_eq!(kline.timestamp, 1704067200000);
        assert_eq!(kline.open, 42000.0);
        assert_eq!(kline.high, 42500.0);
        assert_eq!(kline.low, 41800.0);
        assert_eq!(kline.close, 42300.0);
        assert_eq!(kline.volume, 1000.5);
    }

    #[test]
    fn test_kline_methods() {
        let kline = Kline {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert_eq!(kline.range(), 15.0);
        assert_eq!(kline.body(), 5.0);
        assert!(kline.is_bullish());
    }

    #[test]
    fn test_orderbook_methods() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel {
                    price: 100.0,
                    quantity: 10.0,
                },
                OrderBookLevel {
                    price: 99.0,
                    quantity: 20.0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 101.0,
                    quantity: 15.0,
                },
                OrderBookLevel {
                    price: 102.0,
                    quantity: 25.0,
                },
            ],
        };

        assert_eq!(orderbook.mid_price(), Some(100.5));
        assert_eq!(orderbook.spread(), Some(1.0));
        assert!((orderbook.spread_pct().unwrap() - 0.995).abs() < 0.01);

        // Imbalance: (10+20 - 15+25) / (10+20+15+25) = -10/70
        let imb = orderbook.imbalance(2);
        assert!((imb - (-10.0 / 70.0)).abs() < 0.001);
    }
}
