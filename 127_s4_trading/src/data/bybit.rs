//! Bybit API client for fetching cryptocurrency market data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::{S4Error, Result};

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
    /// Convert candle to feature vector [open, high, low, close, volume].
    pub fn to_features(&self) -> Vec<f64> {
        vec![self.open, self.high, self.low, self.close, self.volume]
    }

    /// Calculate returns from previous close.
    pub fn returns(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close - prev_close) / prev_close
        } else {
            0.0
        }
    }
}

/// Market data container.
#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub interval: String,
    pub candles: Vec<Candle>,
    pub source: String,
}

impl MarketData {
    /// Get close prices.
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get returns series.
    pub fn returns(&self) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..self.candles.len() {
            returns.push(self.candles[i].returns(self.candles[i - 1].close));
        }
        returns
    }

    /// Get feature matrix for model input.
    pub fn to_features(&self) -> Vec<Vec<f64>> {
        self.candles.iter().map(|c| c.to_features()).collect()
    }

    /// Calculate rolling volatility.
    pub fn volatility(&self, window: usize) -> Vec<f64> {
        let returns = self.returns();
        let mut volatility = vec![0.0; returns.len()];

        for i in window..returns.len() {
            let slice = &returns[i - window..i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            volatility[i] = variance.sqrt();
        }

        volatility
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

    /// Fetch kline (OHLCV) data from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "60" for 1 hour)
    /// * `limit` - Number of data points (max 1000)
    ///
    /// # Returns
    /// MarketData with OHLCV candles
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<MarketData> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(1000)
        );

        match self.client.get(&url).send().await {
            Ok(response) => {
                let data: BybitResponse = response.json().await?;

                if data.ret_code != 0 {
                    tracing::warn!("Bybit API error: {}", data.ret_msg);
                    return Ok(self.generate_synthetic(symbol, interval, limit));
                }

                let candles: Vec<Candle> = data
                    .result
                    .list
                    .into_iter()
                    .filter_map(|row| {
                        if row.len() >= 6 {
                            let timestamp_ms: i64 = row[0].parse().ok()?;
                            let timestamp = DateTime::from_timestamp_millis(timestamp_ms)?;
                            Some(Candle {
                                timestamp,
                                open: row[1].parse().ok()?,
                                high: row[2].parse().ok()?,
                                low: row[3].parse().ok()?,
                                close: row[4].parse().ok()?,
                                volume: row[5].parse().ok()?,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                let mut candles = candles;
                candles.reverse(); // Bybit returns newest first

                tracing::info!("Fetched {} klines for {} from Bybit", candles.len(), symbol);

                Ok(MarketData {
                    symbol: symbol.to_string(),
                    interval: interval.to_string(),
                    candles,
                    source: "bybit".to_string(),
                })
            }
            Err(e) => {
                tracing::warn!("Failed to fetch from Bybit: {}. Using synthetic data.", e);
                Ok(self.generate_synthetic(symbol, interval, limit))
            }
        }
    }

    /// Generate synthetic data when API is unavailable.
    pub fn generate_synthetic(&self, symbol: &str, interval: &str, n_points: usize) -> MarketData {
        generate_synthetic_data(symbol, interval, n_points)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic OHLCV data for testing.
pub fn generate_synthetic_data(symbol: &str, interval: &str, n_points: usize) -> MarketData {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.02).unwrap();

    let base_price = if symbol.contains("BTC") {
        50000.0
    } else if symbol.contains("ETH") {
        3000.0
    } else {
        100.0
    };

    let mut candles = Vec::with_capacity(n_points);
    let mut price = base_price;

    let start_time = Utc::now() - chrono::Duration::hours(n_points as i64);

    for i in 0..n_points {
        let return_ = normal.sample(&mut rng);
        price *= 1.0 + return_;

        let noise = rng.gen_range(0.0..0.01);
        let high = price * (1.0 + noise);
        let low = price * (1.0 - noise);
        let open = if i == 0 { base_price } else { candles[i - 1].close };
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: start_time + chrono::Duration::hours(i as i64),
            open,
            high,
            low,
            close: price,
            volume,
        });
    }

    tracing::info!("Generated {} synthetic data points for {}", n_points, symbol);

    MarketData {
        symbol: symbol.to_string(),
        interval: interval.to_string(),
        candles,
        source: "synthetic".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data("BTCUSDT", "60", 100);
        assert_eq!(data.candles.len(), 100);
        assert_eq!(data.symbol, "BTCUSDT");
    }

    #[test]
    fn test_market_data_features() {
        let data = generate_synthetic_data("ETHUSDT", "60", 50);
        let features = data.to_features();
        assert_eq!(features.len(), 50);
        assert_eq!(features[0].len(), 5);
    }

    #[test]
    fn test_returns() {
        let data = generate_synthetic_data("BTCUSDT", "60", 10);
        let returns = data.returns();
        assert_eq!(returns.len(), 10);
        assert_eq!(returns[0], 0.0);
    }
}
