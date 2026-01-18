//! Data Loading and Processing
//!
//! This module provides utilities for loading financial data,
//! including integration with the Bybit cryptocurrency exchange.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during data operations
#[derive(Error, Debug)]
pub enum DataError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("API error: {0}")]
    ApiError(String),
}

/// OHLCV (Open, High, Low, Close, Volume) candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Calculate the return from this candle
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body size (|close - open|)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Bybit API response structures
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
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit data loader for cryptocurrency data
///
/// Fetches historical kline (candlestick) data from Bybit exchange.
///
/// # Example
///
/// ```rust,no_run
/// use positional_encoding::BybitDataLoader;
///
/// let loader = BybitDataLoader::new();
/// let candles = loader.fetch_klines("BTCUSDT", "60", 1000).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BybitDataLoader {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl BybitDataLoader {
    /// Create a new Bybit data loader
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create with custom base URL (for testnet)
    pub fn with_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W)
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Candle structs, ordered from oldest to newest
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, DataError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(1000)
        );

        let response: BybitResponse = self.client.get(&url).send()?.json()?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError(response.ret_msg));
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to oldest first
        candles.reverse();

        Ok(candles)
    }

    /// Fetch multiple pages of klines
    pub fn fetch_klines_extended(
        &self,
        symbol: &str,
        interval: &str,
        total_limit: usize,
    ) -> Result<Vec<Candle>, DataError> {
        let mut all_candles = Vec::new();
        let mut end_time: Option<i64> = None;
        let page_size = 1000;

        while all_candles.len() < total_limit {
            let remaining = total_limit - all_candles.len();
            let limit = remaining.min(page_size);

            let url = match end_time {
                Some(end) => format!(
                    "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}&end={}",
                    self.base_url, symbol, interval, limit, end
                ),
                None => format!(
                    "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
                    self.base_url, symbol, interval, limit
                ),
            };

            let response: BybitResponse = self.client.get(&url).send()?.json()?;

            if response.ret_code != 0 {
                return Err(DataError::ApiError(response.ret_msg));
            }

            if response.result.list.is_empty() {
                break;
            }

            let mut candles: Vec<Candle> = response
                .result
                .list
                .iter()
                .filter_map(|item| {
                    if item.len() >= 6 {
                        Some(Candle {
                            timestamp: item[0].parse().ok()?,
                            open: item[1].parse().ok()?,
                            high: item[2].parse().ok()?,
                            low: item[3].parse().ok()?,
                            close: item[4].parse().ok()?,
                            volume: item[5].parse().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            if candles.is_empty() {
                break;
            }

            // Update end_time for next page (oldest candle timestamp - 1)
            end_time = Some(candles.last().unwrap().timestamp - 1);

            candles.reverse();
            all_candles.splice(0..0, candles);

            if all_candles.len() >= total_limit {
                break;
            }
        }

        // Trim to exact limit if needed
        if all_candles.len() > total_limit {
            all_candles = all_candles[all_candles.len() - total_limit..].to_vec();
        }

        Ok(all_candles)
    }
}

impl Default for BybitDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature preparation utilities
pub struct FeaturePreparator;

impl FeaturePreparator {
    /// Calculate returns from price series
    pub fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 {
                    (w[1] - w[0]) / w[0]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate log returns
    pub fn calculate_log_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 {
                    (w[1] / w[0]).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate rolling mean
    pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![];
        }

        data.windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![];
        }

        data.windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / window as f64;
                let variance = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Normalize data using z-score
    pub fn normalize_zscore(data: &[f64]) -> (Vec<f64>, f64, f64) {
        if data.is_empty() {
            return (vec![], 0.0, 1.0);
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

        let normalized = if std > 1e-10 {
            data.iter().map(|x| (x - mean) / std).collect()
        } else {
            data.iter().map(|x| x - mean).collect()
        };

        (normalized, mean, std)
    }

    /// Prepare features from candles
    pub fn prepare_features(candles: &[Candle]) -> Array2<f64> {
        if candles.is_empty() {
            return Array2::zeros((0, 6));
        }

        let n = candles.len();
        let mut features = Array2::zeros((n, 6));

        // Calculate returns first
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let returns = Self::calculate_returns(&closes);

        for (i, candle) in candles.iter().enumerate() {
            // Return (0 for first candle)
            features[[i, 0]] = if i > 0 { returns[i - 1] } else { 0.0 };

            // Log volume (normalized)
            features[[i, 1]] = if candle.volume > 0.0 {
                candle.volume.ln()
            } else {
                0.0
            };

            // High-Low range relative to close
            features[[i, 2]] = if candle.close > 0.0 {
                (candle.high - candle.low) / candle.close
            } else {
                0.0
            };

            // Body relative to range
            features[[i, 3]] = if candle.range() > 0.0 {
                candle.body() / candle.range()
            } else {
                0.0
            };

            // Direction (1 for bullish, -1 for bearish)
            features[[i, 4]] = if candle.is_bullish() { 1.0 } else { -1.0 };

            // Open-close relative change
            features[[i, 5]] = candle.return_pct();
        }

        features
    }
}

/// Sequence creation for time series models
pub struct SequenceCreator {
    sequence_length: usize,
    target_length: usize,
}

impl SequenceCreator {
    /// Create a new sequence creator
    pub fn new(sequence_length: usize, target_length: usize) -> Self {
        Self {
            sequence_length,
            target_length,
        }
    }

    /// Create sequences from features and targets
    ///
    /// Returns (sequences, targets, timestamps)
    pub fn create_sequences(
        &self,
        features: &Array2<f64>,
        targets: &[f64],
        timestamps: &[i64],
    ) -> (Vec<Array2<f64>>, Vec<Array1<f64>>, Vec<Vec<i64>>) {
        let n_samples = features.nrows();
        if n_samples < self.sequence_length + self.target_length {
            return (vec![], vec![], vec![]);
        }

        let n_sequences = n_samples - self.sequence_length - self.target_length + 1;
        let mut sequences = Vec::with_capacity(n_sequences);
        let mut target_arrays = Vec::with_capacity(n_sequences);
        let mut ts_arrays = Vec::with_capacity(n_sequences);

        for i in 0..n_sequences {
            // Input sequence
            let seq_start = i;
            let seq_end = i + self.sequence_length;
            let seq = features.slice(ndarray::s![seq_start..seq_end, ..]).to_owned();
            sequences.push(seq);

            // Target values
            let target_start = seq_end;
            let target_end = target_start + self.target_length;
            let target = Array1::from_vec(targets[target_start..target_end].to_vec());
            target_arrays.push(target);

            // Timestamps for the sequence
            let ts = timestamps[seq_start..seq_end].to_vec();
            ts_arrays.push(ts);
        }

        (sequences, target_arrays, ts_arrays)
    }

    /// Split data into train and test sets
    pub fn train_test_split<T: Clone>(data: &[T], train_ratio: f64) -> (Vec<T>, Vec<T>) {
        let split_idx = (data.len() as f64 * train_ratio) as usize;
        (data[..split_idx].to_vec(), data[split_idx..].to_vec())
    }
}

/// Generate synthetic data for testing
pub fn generate_synthetic_data(n_samples: usize, seed: u64) -> Vec<Candle> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut candles = Vec::with_capacity(n_samples);

    let base_time = 1704067200i64; // 2024-01-01 00:00:00 UTC
    let interval = 3600i64; // 1 hour

    let mut price = 45000.0; // Starting price

    for i in 0..n_samples {
        // Random walk with trend and volatility
        let trend = 0.0001; // Slight upward trend
        let volatility = 0.02;

        let return_pct = rng.gen::<f64>() * volatility - volatility / 2.0 + trend;
        let open = price;
        price *= 1.0 + return_pct;
        let close = price;

        // Generate high and low
        let range = (close - open).abs() * (1.0 + rng.gen::<f64>() * 0.5);
        let (high, low) = if close > open {
            (
                close + rng.gen::<f64>() * range * 0.3,
                open - rng.gen::<f64>() * range * 0.3,
            )
        } else {
            (
                open + rng.gen::<f64>() * range * 0.3,
                close - rng.gen::<f64>() * range * 0.3,
            )
        };

        // Generate volume with some variance
        let base_volume = 1000.0;
        let volume = base_volume * (0.5 + rng.gen::<f64>() * 1.5);

        candles.push(Candle {
            timestamp: base_time + i as i64 * interval,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_return() {
        let candle = Candle {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
        };

        assert!((candle.return_pct() - 0.05).abs() < 1e-10);
        assert!(candle.is_bullish());
        assert!((candle.range() - 15.0).abs() < 1e-10);
        assert!((candle.body() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.5];
        let returns = FeaturePreparator::calculate_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.10).abs() < 1e-10);
        assert!((returns[1] - (-0.0454545)).abs() < 1e-5);
        assert!((returns[2] - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = FeaturePreparator::rolling_mean(&data, 3);

        assert_eq!(mean.len(), 3);
        assert!((mean[0] - 2.0).abs() < 1e-10);
        assert!((mean[1] - 3.0).abs() < 1e-10);
        assert!((mean[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (normalized, mean, std) = FeaturePreparator::normalize_zscore(&data);

        assert!((mean - 3.0).abs() < 1e-10);
        assert!(normalized[2].abs() < 1e-10); // Middle value should be ~0
    }

    #[test]
    fn test_synthetic_data() {
        let candles = generate_synthetic_data(100, 42);

        assert_eq!(candles.len(), 100);
        assert!(candles[0].timestamp < candles[99].timestamp);

        for candle in &candles {
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open.max(candle.close));
            assert!(candle.low <= candle.open.min(candle.close));
            assert!(candle.volume > 0.0);
        }
    }

    #[test]
    fn test_sequence_creator() {
        let candles = generate_synthetic_data(50, 42);
        let features = FeaturePreparator::prepare_features(&candles);
        let targets: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();
        let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp).collect();

        let creator = SequenceCreator::new(10, 1);
        let (sequences, target_arrays, ts_arrays) =
            creator.create_sequences(&features, &targets, &timestamps);

        assert_eq!(sequences.len(), 40);
        assert_eq!(target_arrays.len(), 40);
        assert_eq!(ts_arrays.len(), 40);

        for seq in &sequences {
            assert_eq!(seq.shape(), &[10, 6]);
        }
    }
}
