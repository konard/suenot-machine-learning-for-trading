//! Bybit API client for cryptocurrency data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Create a new candle.
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

    /// Get OHLCV as a vector.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.open, self.high, self.low, self.close, self.volume]
    }
}

/// API response structures.
#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: ApiResult,
}

#[derive(Debug, Deserialize)]
struct ApiResult {
    list: Vec<Vec<String>>,
}

/// Bybit API client for fetching market data.
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    symbol: String,
    interval: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new(symbol: &str, interval: &str) -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            symbol: symbol.to_string(),
            interval: interval.to_string(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn fetch_klines(&self, limit: usize) -> Result<Vec<Candle>, String> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, self.symbol, self.interval, limit
        );

        let response = reqwest::get(&url)
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        let api_response: ApiResponse = response
            .json()
            .await
            .map_err(|e| format!("JSON parse failed: {}", e))?;

        if api_response.ret_code != 0 {
            return Err(format!("API error: {}", api_response.ret_msg));
        }

        let mut candles: Vec<Candle> = api_response
            .result
            .list
            .iter()
            .filter_map(|kline| {
                if kline.len() >= 6 {
                    let timestamp_ms: i64 = kline[0].parse().ok()?;
                    let timestamp = DateTime::from_timestamp_millis(timestamp_ms)?;
                    Some(Candle {
                        timestamp,
                        open: kline[1].parse().ok()?,
                        high: kline[2].parse().ok()?,
                        low: kline[3].parse().ok()?,
                        close: kline[4].parse().ok()?,
                        volume: kline[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Generate synthetic candle data for testing.
    pub fn generate_synthetic_data(&self, n_candles: usize) -> Vec<Candle> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut candles = Vec::with_capacity(n_candles);

        let mut price = 50000.0; // Starting price (e.g., BTC)
        let now = Utc::now();

        for i in 0..n_candles {
            // Random walk with drift
            let return_pct = rng.gen_range(-0.02..0.02);
            price *= 1.0 + return_pct;

            let volatility = rng.gen_range(0.005..0.02);
            let high = price * (1.0 + volatility);
            let low = price * (1.0 - volatility);
            let open = price * (1.0 + rng.gen_range(-volatility..volatility));
            let close = price;
            let volume = rng.gen_range(100.0..10000.0);

            let timestamp = now - chrono::Duration::hours((n_candles - i) as i64);

            candles.push(Candle {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        candles
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new("BTCUSDT", "60")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let client = BybitClient::default();
        let candles = client.generate_synthetic_data(100);

        assert_eq!(candles.len(), 100);
        for candle in &candles {
            assert!(candle.high >= candle.low);
            assert!(candle.volume > 0.0);
        }
    }
}
