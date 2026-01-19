//! API type definitions
//!
//! Common types used for market data APIs.

use serde::{Deserialize, Serialize};

/// Kline (candlestick) interval
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KlineInterval {
    #[serde(rename = "1")]
    Min1,
    #[serde(rename = "3")]
    Min3,
    #[serde(rename = "5")]
    Min5,
    #[serde(rename = "15")]
    Min15,
    #[serde(rename = "30")]
    Min30,
    #[serde(rename = "60")]
    Hour1,
    #[serde(rename = "120")]
    Hour2,
    #[serde(rename = "240")]
    Hour4,
    #[serde(rename = "360")]
    Hour6,
    #[serde(rename = "720")]
    Hour12,
    #[serde(rename = "D")]
    Day1,
    #[serde(rename = "W")]
    Week1,
    #[serde(rename = "M")]
    Month1,
}

impl KlineInterval {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Min1 => "1",
            Self::Min3 => "3",
            Self::Min5 => "5",
            Self::Min15 => "15",
            Self::Min30 => "30",
            Self::Hour1 => "60",
            Self::Hour2 => "120",
            Self::Hour4 => "240",
            Self::Hour6 => "360",
            Self::Hour12 => "720",
            Self::Day1 => "D",
            Self::Week1 => "W",
            Self::Month1 => "M",
        }
    }

    /// Convert to minutes for calculations
    pub fn to_minutes(&self) -> u32 {
        match self {
            Self::Min1 => 1,
            Self::Min3 => 3,
            Self::Min5 => 5,
            Self::Min15 => 15,
            Self::Min30 => 30,
            Self::Hour1 => 60,
            Self::Hour2 => 120,
            Self::Hour4 => 240,
            Self::Hour6 => 360,
            Self::Hour12 => 720,
            Self::Day1 => 1440,
            Self::Week1 => 10080,
            Self::Month1 => 43200,
        }
    }
}

impl std::fmt::Display for KlineInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Single kline/candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl KlineData {
    /// Calculate the typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open
        }
    }
}

/// Market ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub change_24h: f64,
    pub change_pct_24h: f64,
}

/// Generic market data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub interval: String,
    pub klines: Vec<KlineData>,
}

impl MarketData {
    /// Create new market data
    pub fn new(symbol: String, interval: String, klines: Vec<KlineData>) -> Self {
        Self {
            symbol,
            interval,
            klines,
        }
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.klines.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.klines.is_empty()
    }

    /// Get close prices as vector
    pub fn close_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.close).collect()
    }

    /// Get volumes as vector
    pub fn volumes(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.volume).collect()
    }

    /// Calculate simple returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }
}

/// API response wrapper for Bybit
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Bybit kline result
#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_data() {
        let kline = KlineData {
            timestamp: 1234567890000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(kline.is_bullish());
        assert_eq!(kline.body(), 5.0);
        assert_eq!(kline.range(), 15.0);
        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert!((kline.return_pct() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_market_data() {
        let klines = vec![
            KlineData {
                timestamp: 1000,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 100.0,
                turnover: 10000.0,
            },
            KlineData {
                timestamp: 2000,
                open: 102.0,
                high: 108.0,
                low: 100.0,
                close: 106.0,
                volume: 150.0,
                turnover: 15000.0,
            },
        ];

        let data = MarketData::new("BTCUSDT".to_string(), "1h".to_string(), klines);

        assert_eq!(data.len(), 2);
        assert_eq!(data.close_prices(), vec![102.0, 106.0]);

        let returns = data.returns();
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 0.0392).abs() < 0.001);
    }

    #[test]
    fn test_interval() {
        assert_eq!(KlineInterval::Hour1.as_str(), "60");
        assert_eq!(KlineInterval::Day1.to_minutes(), 1440);
    }
}
