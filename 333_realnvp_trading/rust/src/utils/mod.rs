//! # Utility Functions
//!
//! Common utilities for data handling, feature engineering, and normalization.

use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// OHLCV Candle data
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

    /// Calculate return from open to close
    pub fn return_(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let hl = self.high - self.low;
        match prev_close {
            Some(pc) => {
                let hc = (self.high - pc).abs();
                let lc = (self.low - pc).abs();
                hl.max(hc).max(lc)
            }
            None => hl,
        }
    }

    /// Get typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Market state features
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Multi-scale returns
    pub returns: Vec<f64>,
    /// Volatility
    pub volatility: f64,
    /// Volatility ratio
    pub volatility_ratio: f64,
    /// Momentum
    pub momentum: f64,
    /// Volume ratio
    pub volume_ratio: f64,
    /// Price position
    pub price_position: f64,
    /// RSI-like indicator
    pub rsi: f64,
}

impl MarketState {
    /// Convert to array
    pub fn to_array(&self) -> Array1<f64> {
        let mut features = self.returns.clone();
        features.extend([
            self.volatility,
            self.volatility_ratio,
            self.momentum,
            self.volume_ratio,
            self.price_position,
            self.rsi,
        ]);
        Array1::from_vec(features)
    }
}

/// Compute market state from candles
pub fn compute_market_state(candles: &[Candle]) -> Array1<f64> {
    if candles.is_empty() {
        return Array1::zeros(10);
    }

    let lookback = candles.len();
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // Calculate returns
    let returns: Vec<f64> = closes.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let mut features = Vec::new();

    // Multi-scale returns
    for period in [1, 5, 10, 20] {
        if returns.len() >= period {
            let sum: f64 = returns[returns.len() - period..].iter().sum();
            features.push(sum);
        } else if !returns.is_empty() {
            features.push(returns.iter().sum());
        } else {
            features.push(0.0);
        }
    }

    // Volatility (standard deviation of returns)
    let volatility = if !returns.is_empty() {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    } else {
        0.0
    };
    features.push(volatility);

    // Volatility ratio (short-term / long-term)
    let short_vol = if returns.len() >= 5 {
        let short_returns = &returns[returns.len() - 5..];
        let mean = short_returns.iter().sum::<f64>() / 5.0;
        let variance = short_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / 5.0;
        variance.sqrt()
    } else {
        volatility
    };
    let volatility_ratio = if volatility > 1e-10 {
        short_vol / volatility
    } else {
        1.0
    };
    features.push(volatility_ratio);

    // Momentum
    let momentum = if closes.len() >= 2 {
        (closes.last().unwrap() / closes.first().unwrap()) - 1.0
    } else {
        0.0
    };
    features.push(momentum);

    // Volume ratio
    let volume_mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let volume_ratio = if volume_mean > 0.0 {
        volumes.last().unwrap_or(&volume_mean) / volume_mean
    } else {
        1.0
    };
    features.push(volume_ratio);

    // Price position (current price relative to range)
    let high_20 = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low_20 = lows.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = high_20 - low_20;
    let price_position = if range > 1e-10 {
        (closes.last().unwrap() - low_20) / range
    } else {
        0.5
    };
    features.push(price_position);

    // RSI-like indicator
    let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let rsi = if gains + losses > 0.0 {
        gains / (gains + losses)
    } else {
        0.5
    };
    features.push(rsi);

    Array1::from_vec(features)
}

/// Normalize features using z-score
pub fn normalize_features(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mean = features.mean_axis(Axis(0)).unwrap();
    let std = features.std_axis(Axis(0), 0.0);

    // Add small epsilon to avoid division by zero
    let std_safe: Array1<f64> = std.mapv(|s| if s < 1e-8 { 1.0 } else { s });

    let normalized = (features - &mean) / &std_safe;

    (normalized, mean, std_safe)
}

/// Denormalize features
pub fn denormalize_features(
    normalized: &Array2<f64>,
    mean: &Array1<f64>,
    std: &Array1<f64>,
) -> Array2<f64> {
    normalized * std + mean
}

/// Calculate rolling statistics
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        return vec![data.iter().sum::<f64>() / data.len() as f64; data.len()];
    }

    data.windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

/// Calculate rolling standard deviation
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        return vec![variance.sqrt(); data.len()];
    }

    data.windows(window)
        .map(|w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            let variance = w.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            variance.sqrt()
        })
        .collect()
}

/// Clip values to specified range
pub fn clip_outliers(data: &mut Array1<f64>, n_std: f64) {
    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(0.0);
    let lower = mean - n_std * std;
    let upper = mean + n_std * std;

    data.mapv_inplace(|x| x.clamp(lower, upper));
}

/// Simple train/validation/test split
pub fn train_val_test_split(
    data: &Array2<f64>,
    train_ratio: f64,
    val_ratio: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n = data.nrows();
    let train_end = (n as f64 * train_ratio) as usize;
    let val_end = train_end + (n as f64 * val_ratio) as usize;

    let train = data.slice(ndarray::s![..train_end, ..]).to_owned();
    let val = data.slice(ndarray::s![train_end..val_end, ..]).to_owned();
    let test = data.slice(ndarray::s![val_end.., ..]).to_owned();

    (train, val, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_return() {
        let candle = Candle {
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 102.0,
            volume: 1000.0,
        };

        assert!((candle.return_() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_compute_market_state() {
        let candles: Vec<Candle> = (0..20).map(|i| {
            Candle {
                timestamp: Utc::now(),
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            }
        }).collect();

        let state = compute_market_state(&candles);
        assert_eq!(state.len(), 10);
        assert!(state.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_normalize_features() {
        let features = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ).unwrap();

        let (normalized, mean, std) = normalize_features(&features);

        // Check mean is approximately zero
        let normalized_mean = normalized.mean_axis(Axis(0)).unwrap();
        for &m in normalized_mean.iter() {
            assert!(m.abs() < 1e-10);
        }
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_train_val_test_split() {
        let data = Array2::from_shape_vec(
            (100, 5),
            (0..500).map(|x| x as f64).collect(),
        ).unwrap();

        let (train, val, test) = train_val_test_split(&data, 0.7, 0.15);

        assert_eq!(train.nrows(), 70);
        assert_eq!(val.nrows(), 15);
        assert_eq!(test.nrows(), 15);
    }
}
