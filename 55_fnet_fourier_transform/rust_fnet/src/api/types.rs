//! API response types for Bybit.

use serde::{Deserialize, Serialize};

/// Kline (candlestick) data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Timestamp in milliseconds
    pub timestamp: i64,
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
    /// Turnover
    pub turnover: f64,
}

/// Bybit API response for klines.
#[derive(Debug, Deserialize)]
pub struct KlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: KlineResult,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

impl KlineResult {
    /// Parse raw list data into Kline structs.
    pub fn parse_klines(&self) -> Vec<Kline> {
        self.list
            .iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Market info response.
#[derive(Debug, Deserialize)]
pub struct MarketInfoResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: MarketInfoResult,
}

#[derive(Debug, Deserialize)]
pub struct MarketInfoResult {
    pub category: String,
    pub list: Vec<MarketInfo>,
}

#[derive(Debug, Deserialize)]
pub struct MarketInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_parsing() {
        let result = KlineResult {
            symbol: "BTCUSDT".to_string(),
            category: "linear".to_string(),
            list: vec![
                vec![
                    "1704067200000".to_string(),
                    "42000.5".to_string(),
                    "42500.0".to_string(),
                    "41800.0".to_string(),
                    "42200.0".to_string(),
                    "1234.5".to_string(),
                    "51897000".to_string(),
                ],
            ],
        };

        let klines = result.parse_klines();
        assert_eq!(klines.len(), 1);
        assert_eq!(klines[0].timestamp, 1704067200000);
        assert!((klines[0].open - 42000.5).abs() < 0.01);
    }
}
