//! Bybit API client for market data.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Bybit API client.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval in minutes (e.g., "60" for 1 hour)
    /// * `limit` - Number of candles to fetch
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<KlineData>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse = response.json().await?;

        if data.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", data.ret_msg);
        }

        let mut klines: Vec<KlineData> = data.result.list
            .into_iter()
            .map(|k| KlineData {
                timestamp: k.0.parse::<i64>().unwrap_or(0),
                open: k.1.parse::<f64>().unwrap_or(0.0),
                high: k.2.parse::<f64>().unwrap_or(0.0),
                low: k.3.parse::<f64>().unwrap_or(0.0),
                close: k.4.parse::<f64>().unwrap_or(0.0),
                volume: k.5.parse::<f64>().unwrap_or(0.0),
                turnover: k.6.parse::<f64>().unwrap_or(0.0),
            })
            .collect();

        // Sort by timestamp (API returns newest first)
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Calculate features from kline data.
    pub fn calculate_features(klines: &[KlineData]) -> Vec<MarketData> {
        if klines.len() < 25 {
            return Vec::new();
        }

        let mut features = Vec::with_capacity(klines.len());

        for i in 24..klines.len() {
            let current = &klines[i];
            let prev = &klines[i - 1];

            // Log return
            let log_return = (current.close / prev.close).ln();

            // Volatility (rolling std of returns)
            let returns: Vec<f64> = (i - 23..=i)
                .map(|j| (klines[j].close / klines[j - 1].close).ln())
                .collect();
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let volatility = (returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64)
                .sqrt();

            // Volume ratio
            let avg_volume: f64 = (i - 23..=i)
                .map(|j| klines[j].volume)
                .sum::<f64>() / 24.0;
            let volume_ratio = current.volume / (avg_volume + 1e-8);

            // Momentum (24-period return)
            let momentum = current.close / klines[i - 24].close - 1.0;

            // RSI (14-period)
            let rsi = Self::calculate_rsi(&klines[..=i], 14);

            features.push(MarketData {
                timestamp: current.timestamp,
                close: current.close,
                log_return,
                volatility,
                volume_ratio,
                momentum,
                rsi,
            });
        }

        features
    }

    /// Calculate RSI indicator.
    fn calculate_rsi(klines: &[KlineData], period: usize) -> f64 {
        if klines.len() < period + 1 {
            return 50.0;
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        let start = klines.len() - period - 1;
        for i in start..klines.len() - 1 {
            let change = klines[i + 1].close - klines[i].close;
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let avg_gain: f64 = gains.iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses.iter().sum::<f64>() / period as f64;

        if avg_loss < 1e-8 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Bybit API response structure.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<(String, String, String, String, String, String, String)>,
}

/// Kline (candlestick) data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
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

/// Processed market data with features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Close price
    pub close: f64,
    /// Log return
    pub log_return: f64,
    /// Volatility (24-period rolling std)
    pub volatility: f64,
    /// Volume ratio (current / 24-period avg)
    pub volume_ratio: f64,
    /// Momentum (24-period return)
    pub momentum: f64,
    /// RSI (14-period)
    pub rsi: f64,
}

impl MarketData {
    /// Convert to feature vector for model input.
    pub fn to_features(&self) -> Vec<f32> {
        vec![
            self.log_return as f32,
            self.volatility as f32,
            self.volume_ratio as f32,
            self.momentum as f32,
            self.rsi as f32 / 100.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        // Create synthetic kline data
        let klines: Vec<KlineData> = (0..20)
            .map(|i| KlineData {
                timestamp: i as i64 * 60000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.0 + i as f64,  // Steadily increasing
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect();

        let rsi = BybitClient::calculate_rsi(&klines, 14);

        // With steadily increasing prices, RSI should be high
        assert!(rsi > 50.0, "RSI should be > 50 for increasing prices: {}", rsi);
    }

    #[test]
    fn test_market_data_features() {
        let data = MarketData {
            timestamp: 0,
            close: 50000.0,
            log_return: 0.001,
            volatility: 0.02,
            volume_ratio: 1.5,
            momentum: 0.05,
            rsi: 60.0,
        };

        let features = data.to_features();
        assert_eq!(features.len(), 5);
        assert!((features[4] - 0.6).abs() < 0.001); // RSI normalized to 0-1
    }
}
