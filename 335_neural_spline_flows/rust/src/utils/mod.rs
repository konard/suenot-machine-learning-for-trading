//! Utility functions for data processing and feature engineering
//!
//! This module provides utilities for working with market data:
//! - Candle data structures
//! - Feature extraction
//! - Data normalization

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
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
}

impl Candle {
    /// Create a new candle
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

    /// Calculate the typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Feature vector extracted from market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Feature values
    pub values: Vec<f64>,
    /// Feature names
    pub names: Vec<String>,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(values: Vec<f64>, names: Vec<String>) -> Self {
        assert_eq!(values.len(), names.len());
        Self { values, names }
    }

    /// Get feature by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }

    /// Number of features
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Convert to ndarray
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(self.values.clone())
    }
}

/// Extract features from candle data
///
/// Features extracted:
/// - Returns at multiple scales (1, 5, 10, 20 periods)
/// - Volatility measures
/// - Volume patterns
/// - Price position in range
/// - Momentum indicators
pub fn extract_features(candles: &[Candle]) -> FeatureVector {
    let n = candles.len();
    if n < 20 {
        // Return empty features for insufficient data
        return FeatureVector::new(vec![0.0; 10], vec!["empty".to_string(); 10]);
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();

    let mut values = Vec::new();
    let mut names = Vec::new();

    // Returns at multiple scales
    for period in [1, 5, 10, 20] {
        let ret = if n > period {
            (closes[n - 1] / closes[n - 1 - period] - 1.0)
        } else {
            0.0
        };
        values.push(ret);
        names.push(format!("return_{}", period));
    }

    // Volatility (20-period)
    let returns: Vec<f64> = (1..n)
        .map(|i| closes[i] / closes[i - 1] - 1.0)
        .collect();

    let volatility = if returns.len() >= 20 {
        let recent = &returns[returns.len() - 20..];
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        variance.sqrt()
    } else {
        0.01
    };
    values.push(volatility);
    names.push("volatility_20".to_string());

    // Short-term volatility (5-period)
    let vol_5 = if returns.len() >= 5 {
        let recent = &returns[returns.len() - 5..];
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        variance.sqrt()
    } else {
        volatility
    };
    values.push(vol_5);
    names.push("volatility_5".to_string());

    // Volatility ratio
    let vol_ratio = if volatility > 0.0 { vol_5 / volatility } else { 1.0 };
    values.push(vol_ratio);
    names.push("vol_ratio".to_string());

    // Volume ratio
    let vol_ma: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let vol_ratio = if vol_ma > 0.0 {
        volumes[n - 1] / vol_ma
    } else {
        1.0
    };
    values.push(vol_ratio);
    names.push("volume_ratio".to_string());

    // Price position in 20-period range
    let high_20 = highs.iter().rev().take(20).cloned().fold(f64::NEG_INFINITY, f64::max);
    let low_20 = lows.iter().rev().take(20).cloned().fold(f64::INFINITY, f64::min);
    let price_position = if high_20 > low_20 {
        (closes[n - 1] - low_20) / (high_20 - low_20)
    } else {
        0.5
    };
    values.push(price_position);
    names.push("price_position".to_string());

    // RSI-like momentum
    let rsi = compute_rsi(&closes, 14);
    values.push(rsi);
    names.push("rsi".to_string());

    FeatureVector::new(values, names)
}

/// Compute RSI indicator
fn compute_rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 0.5;
    }

    let mut gains = 0.0;
    let mut losses = 0.0;

    for i in (closes.len() - period)..closes.len() {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        return 1.0;
    }

    let rs = avg_gain / avg_loss;
    1.0 - 1.0 / (1.0 + rs)
}

/// Normalize feature vectors to zero mean and unit variance
pub fn normalize_features(features: &[FeatureVector]) -> (Vec<FeatureVector>, Array1<f64>, Array1<f64>) {
    if features.is_empty() {
        return (vec![], Array1::zeros(0), Array1::ones(0));
    }

    let dim = features[0].len();
    let n = features.len() as f64;

    // Compute mean
    let mut mean = vec![0.0; dim];
    for f in features {
        for (i, v) in f.values.iter().enumerate() {
            mean[i] += v;
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    // Compute std
    let mut variance = vec![0.0; dim];
    for f in features {
        for (i, v) in f.values.iter().enumerate() {
            variance[i] += (v - mean[i]).powi(2);
        }
    }
    let std: Vec<f64> = variance.iter().map(|v| (v / n).sqrt().max(1e-6)).collect();

    // Normalize
    let normalized: Vec<FeatureVector> = features
        .iter()
        .map(|f| {
            let values: Vec<f64> = f
                .values
                .iter()
                .zip(mean.iter().zip(std.iter()))
                .map(|(v, (m, s))| (v - m) / s)
                .collect();
            FeatureVector::new(values, f.names.clone())
        })
        .collect();

    (
        normalized,
        Array1::from_vec(mean),
        Array1::from_vec(std),
    )
}

/// Convert candles to feature matrix
pub fn candles_to_features(candles: &[Candle], lookback: usize) -> Vec<FeatureVector> {
    let mut features = Vec::new();

    for i in lookback..candles.len() {
        let window = &candles[(i - lookback)..i];
        features.push(extract_features(window));
    }

    features
}

/// Statistics helper
pub struct Statistics;

impl Statistics {
    /// Calculate mean
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    pub fn std(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(data);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    /// Calculate median
    pub fn median(data: &mut [f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = data.len() / 2;
        if data.len() % 2 == 0 {
            (data[mid - 1] + data[mid]) / 2.0
        } else {
            data[mid]
        }
    }

    /// Calculate percentile
    pub fn percentile(data: &mut [f64], p: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (data.len() - 1) as f64) as usize;
        data[idx.min(data.len() - 1)]
    }

    /// Calculate skewness
    pub fn skewness(data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }
        let mean = Self::mean(data);
        let std = Self::std(data);
        if std == 0.0 {
            return 0.0;
        }
        let n = data.len() as f64;
        data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n
    }

    /// Calculate kurtosis (excess kurtosis)
    pub fn kurtosis(data: &[f64]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }
        let mean = Self::mean(data);
        let std = Self::std(data);
        if std == 0.0 {
            return 0.0;
        }
        let n = data.len() as f64;
        data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n - 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles() -> Vec<Candle> {
        let now = Utc::now();
        (0..30)
            .map(|i| {
                let price = 100.0 + i as f64;
                Candle {
                    timestamp: now + chrono::Duration::hours(i as i64),
                    open: price - 0.5,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_candle() {
        let candle = Candle::new(Utc::now(), 100.0, 105.0, 95.0, 102.0, 1000.0);
        assert!(candle.is_bullish());
        assert_eq!(candle.range(), 10.0);
    }

    #[test]
    fn test_extract_features() {
        let candles = create_test_candles();
        let features = extract_features(&candles);

        assert!(!features.is_empty());
        assert!(features.get("return_1").is_some());
        assert!(features.get("volatility_20").is_some());
    }

    #[test]
    fn test_normalize_features() {
        let candles = create_test_candles();
        let features: Vec<FeatureVector> = (20..30)
            .map(|i| extract_features(&candles[..i]))
            .collect();

        let (normalized, mean, std) = normalize_features(&features);

        assert_eq!(normalized.len(), features.len());
        assert_eq!(mean.len(), features[0].len());
        assert_eq!(std.len(), features[0].len());
    }

    #[test]
    fn test_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(Statistics::mean(&data), 3.0);
        assert!(Statistics::std(&data) > 0.0);
        assert_eq!(Statistics::median(&mut data.clone()), 3.0);
    }
}
