//! Data loading module for LLM Alpha Mining.
//!
//! Provides utilities for loading market data from various sources
//! including Bybit for cryptocurrency data.

use crate::error::{Error, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OHLCV candle data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Market data container.
#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub source: String,
    pub candles: Vec<OHLCV>,
}

impl MarketData {
    /// Create new market data container.
    pub fn new(symbol: String, source: String, candles: Vec<OHLCV>) -> Self {
        Self { symbol, source, candles }
    }

    /// Get number of candles.
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get close prices as a vector.
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get volumes as a vector.
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Get timestamps as a vector.
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.candles.iter().map(|c| c.timestamp).collect()
    }

    /// Calculate returns.
    /// Returns n elements (same length as close_prices) for vector alignment.
    /// returns[i] = (close[i+1] - close[i]) / close[i] for i < n-1, with NaN at index n-1.
    /// This represents forward returns: returns[i] is the return from period i to period i+1.
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        let mut result: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        result.push(f64::NAN); // Last element is undefined (no future data)
        result
    }

    /// Calculate log returns.
    /// Returns n elements (same length as close_prices) for vector alignment.
    /// log_returns[i] = ln(close[i+1] / close[i]) for i < n-1, with NaN at index n-1.
    /// This represents forward log returns.
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        let mut result: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        result.push(f64::NAN); // Last element is undefined (no future data)
        result
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
    list: Vec<Vec<String>>,
}

/// Bybit data loader.
///
/// Fetches OHLCV data from Bybit's public API.
///
/// # Example
///
/// ```rust,ignore
/// use llm_alpha_mining::data::BybitLoader;
///
/// #[tokio::main]
/// async fn main() {
///     let loader = BybitLoader::new();
///     let data = loader.load("BTCUSDT", "60", 30).await.unwrap();
///     println!("Loaded {} candles", data.len());
/// }
/// ```
pub struct BybitLoader {
    client: reqwest::Client,
    base_url: String,
}

impl BybitLoader {
    /// Create a new Bybit loader.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a new Bybit loader for testnet.
    pub fn testnet() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Load OHLCV data from Bybit.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "30", "60", "240", "D", "W")
    /// * `days` - Number of days of data to fetch
    ///
    /// # Returns
    ///
    /// MarketData containing the OHLCV candles.
    pub async fn load(
        &self,
        symbol: &str,
        interval: &str,
        days: i64,
    ) -> Result<MarketData> {
        let end_time = Utc::now();
        let start_time = end_time - Duration::days(days);

        let end_ts = end_time.timestamp_millis();
        let start_ts = start_time.timestamp_millis();

        let mut all_candles = Vec::new();
        let mut current_end = end_ts;

        while current_end > start_ts {
            let url = format!(
                "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
                self.base_url, symbol, interval, start_ts, current_end
            );

            let response: BybitResponse = self.client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                return Err(Error::Api(response.ret_msg));
            }

            let candles = response.result.list;
            if candles.is_empty() {
                break;
            }

            for candle in &candles {
                if candle.len() >= 7 {
                    let timestamp_ms: i64 = candle[0].parse().unwrap_or(0);
                    let timestamp = DateTime::from_timestamp_millis(timestamp_ms)
                        .unwrap_or(Utc::now());

                    all_candles.push(OHLCV {
                        timestamp,
                        open: candle[1].parse().unwrap_or(0.0),
                        high: candle[2].parse().unwrap_or(0.0),
                        low: candle[3].parse().unwrap_or(0.0),
                        close: candle[4].parse().unwrap_or(0.0),
                        volume: candle[5].parse().unwrap_or(0.0),
                        turnover: candle[6].parse().unwrap_or(0.0),
                    });
                }
            }

            // Get oldest timestamp for pagination
            let oldest = candles
                .iter()
                .filter_map(|c| c.get(0))
                .filter_map(|t| t.parse::<i64>().ok())
                .min()
                .unwrap_or(start_ts);

            current_end = oldest - 1;

            if candles.len() < 1000 {
                break;
            }
        }

        // Sort by timestamp
        all_candles.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Filter to requested range
        let filtered: Vec<OHLCV> = all_candles
            .into_iter()
            .filter(|c| c.timestamp >= start_time)
            .collect();

        if filtered.is_empty() {
            return Err(Error::InsufficientData(format!(
                "No data found for {} in the last {} days",
                symbol, days
            )));
        }

        Ok(MarketData::new(symbol.to_string(), "bybit".to_string(), filtered))
    }

    /// Load funding rate history.
    pub async fn load_funding_rate(
        &self,
        symbol: &str,
        days: i64,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        let end_time = Utc::now();
        let start_time = end_time - Duration::days(days);

        let url = format!(
            "{}/v5/market/funding/history?category=linear&symbol={}&startTime={}&endTime={}&limit=200",
            self.base_url,
            symbol,
            start_time.timestamp_millis(),
            end_time.timestamp_millis()
        );

        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            let msg = response["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::Api(msg.to_string()));
        }

        let list = response["result"]["list"]
            .as_array()
            .ok_or_else(|| Error::Api("Invalid response format".to_string()))?;

        let mut rates = Vec::new();
        for item in list {
            let timestamp_ms: i64 = item["fundingRateTimestamp"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            let rate: f64 = item["fundingRate"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);

            if let Some(ts) = DateTime::from_timestamp_millis(timestamp_ms) {
                rates.push((ts, rate));
            }
        }

        rates.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(rates)
    }
}

impl Default for BybitLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic market data for testing.
///
/// # Arguments
///
/// * `symbol` - Symbol name
/// * `days` - Number of days of data
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// MarketData with synthetic candles.
pub fn generate_synthetic_data(symbol: &str, days: usize, seed: u64) -> MarketData {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    let mut candles = Vec::with_capacity(days);
    let start_time = Utc::now() - Duration::days(days as i64);
    let mut price = 100.0;

    for i in 0..days {
        let return_val: f64 = rng.gen::<f64>() * 0.04 - 0.02; // -2% to +2%
        price *= 1.0 + return_val;

        let volatility = rng.gen::<f64>() * 0.02 + 0.005;
        let open = price * (1.0 + (rng.gen::<f64>() - 0.5) * volatility);
        let high = price * (1.0 + rng.gen::<f64>().abs() * volatility);
        let low = price * (1.0 - rng.gen::<f64>().abs() * volatility);

        candles.push(OHLCV {
            timestamp: start_time + Duration::days(i as i64),
            open,
            high,
            low,
            close: price,
            volume: rng.gen_range(1_000_000.0..100_000_000.0),
            turnover: price * rng.gen_range(1_000_000.0..100_000_000.0),
        });
    }

    MarketData::new(symbol.to_string(), "synthetic".to_string(), candles)
}

/// Calculate technical features from market data.
pub fn calculate_features(data: &MarketData) -> HashMap<String, Vec<f64>> {
    let mut features = HashMap::new();

    let closes = data.close_prices();
    let volumes = data.volumes();
    let n = closes.len();

    // Returns
    let returns = data.returns();
    features.insert("return".to_string(), returns.clone());

    // Log returns
    let log_returns = data.log_returns();
    features.insert("log_return".to_string(), log_returns.clone());

    // Rolling mean (SMA 20)
    let sma_20 = rolling_mean(&closes, 20);
    features.insert("sma_20".to_string(), sma_20);

    // Rolling std (volatility)
    let volatility = rolling_std(&log_returns, 20);
    features.insert("volatility_20".to_string(), volatility);

    // Volume ratio
    let volume_sma = rolling_mean(&volumes, 20);
    let volume_ratio: Vec<f64> = volumes
        .iter()
        .zip(volume_sma.iter())
        .map(|(v, avg)| if *avg > 0.0 { v / avg } else { 1.0 })
        .collect();
    features.insert("volume_ratio".to_string(), volume_ratio);

    // Momentum (5-day)
    let momentum: Vec<f64> = (0..n)
        .map(|i| {
            if i >= 5 {
                closes[i] / closes[i - 5] - 1.0
            } else {
                0.0
            }
        })
        .collect();
    features.insert("momentum_5".to_string(), momentum);

    features
}

/// Calculate rolling mean.
fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let sum: f64 = data[(i + 1 - window)..=i].iter().sum();
        result[i] = sum / window as f64;
    }

    result
}

/// Calculate rolling standard deviation.
/// Uses sample variance (ddof=1) to match pandas default behavior.
fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    // Requires window >= 2 for valid sample std calculation
    if window < 2 {
        return result;
    }

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
        result[i] = variance.sqrt();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data("TEST", 100, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data.symbol, "TEST");
        assert_eq!(data.source, "synthetic");
    }

    #[test]
    fn test_returns() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let returns = data.returns();
        // Returns now has same length as close_prices (n elements)
        assert_eq!(returns.len(), 100);
        // First elements should be valid (forward returns)
        assert!(!returns[0].is_nan());
        // Last element should be NaN (no future data)
        assert!(returns[99].is_nan());
    }

    #[test]
    fn test_features() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let features = calculate_features(&data);
        assert!(features.contains_key("sma_20"));
        assert!(features.contains_key("volatility_20"));
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }
}
