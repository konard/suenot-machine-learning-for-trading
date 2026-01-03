//! # Utilities Module
//!
//! Common utilities for data handling and feature computation.

use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// OHLCV Candle data
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

    /// Calculate typical price
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate candle range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body size
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Is bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Market state representation
#[derive(Debug, Clone)]
pub struct MarketState {
    pub features: Array1<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Compute market features from candle data
///
/// Features:
/// 0. return_1  - 1-period return
/// 1. return_5  - 5-period return
/// 2. return_10 - 10-period return
/// 3. return_20 - 20-period return
/// 4. volatility - Rolling volatility
/// 5. vol_ratio - Short/Long volatility ratio
/// 6. volume_ratio - Volume relative to MA
/// 7. price_position - Position in 20-period range [0, 1]
/// 8. trend - EMA fast/slow difference
pub fn compute_market_features(candles: &[Candle], lookback: usize) -> Array1<f64> {
    let n = candles.len();
    assert!(n >= lookback, "Not enough candles for lookback period");

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // Returns at multiple scales
    let return_1 = if n >= 2 {
        closes[n - 1] / closes[n - 2] - 1.0
    } else {
        0.0
    };

    let return_5 = if n >= 5 {
        closes[n - 1] / closes[n - 5] - 1.0
    } else {
        0.0
    };

    let return_10 = if n >= 10 {
        closes[n - 1] / closes[n - 10] - 1.0
    } else {
        0.0
    };

    let return_20 = if n >= lookback {
        closes[n - 1] / closes[n - lookback] - 1.0
    } else {
        0.0
    };

    // Log returns for volatility calculation
    let log_returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    // Volatility (standard deviation of returns)
    let volatility = if log_returns.len() >= lookback {
        let recent = &log_returns[log_returns.len() - lookback..];
        std_dev(recent)
    } else {
        std_dev(&log_returns)
    };

    // Volatility ratio (short-term / long-term)
    let vol_ratio = if log_returns.len() >= lookback {
        let short_vol = std_dev(&log_returns[log_returns.len() - 5..]);
        let long_vol = std_dev(&log_returns[log_returns.len() - lookback..]);
        if long_vol > 1e-8 {
            short_vol / long_vol
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Volume ratio
    let volume_ma: f64 = volumes[n - lookback..].iter().sum::<f64>() / lookback as f64;
    let volume_ratio = if volume_ma > 1e-8 {
        volumes[n - 1] / volume_ma
    } else {
        1.0
    };

    // Price position in range
    let high_20 = highs[n - lookback..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low_20 = lows[n - lookback..].iter().cloned().fold(f64::INFINITY, f64::min);
    let price_range = high_20 - low_20;
    let price_position = if price_range > 1e-8 {
        (closes[n - 1] - low_20) / price_range
    } else {
        0.5
    };

    // Trend (EMA difference)
    let ema_fast = ema(&closes, 5);
    let ema_slow = ema(&closes, 20);
    let trend = if closes[n - 1] > 1e-8 {
        (ema_fast - ema_slow) / closes[n - 1]
    } else {
        0.0
    };

    Array1::from_vec(vec![
        return_1,
        return_5,
        return_10,
        return_20,
        volatility,
        vol_ratio,
        volume_ratio,
        price_position,
        trend,
    ])
}

/// Compute features for a batch of windows
pub fn compute_features_batch(candles: &[Candle], lookback: usize) -> Array2<f64> {
    let n_samples = candles.len() - lookback + 1;
    let n_features = 9;

    let mut features = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        let window = &candles[i..i + lookback];
        let feats = compute_market_features(window, lookback);
        for (j, &val) in feats.iter().enumerate() {
            features[[i, j]] = val;
        }
    }

    features
}

/// Normalize features using z-score normalization
pub fn normalize_features(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_features = features.ncols();
    let mut means = Array1::zeros(n_features);
    let mut stds = Array1::zeros(n_features);

    // Compute mean and std for each feature
    for j in 0..n_features {
        let col = features.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let std = std_dev(col.as_slice().unwrap());

        means[j] = mean;
        stds[j] = if std > 1e-8 { std } else { 1.0 };
    }

    // Normalize
    let mut normalized = features.clone();
    for j in 0..n_features {
        for i in 0..features.nrows() {
            normalized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
        }
    }

    (normalized, means, stds)
}

/// Apply normalization with pre-computed mean and std
pub fn apply_normalization(
    features: &Array1<f64>,
    means: &Array1<f64>,
    stds: &Array1<f64>,
) -> Array1<f64> {
    let mut normalized = Array1::zeros(features.len());
    for i in 0..features.len() {
        normalized[i] = (features[i] - means[i]) / stds[i];
    }
    normalized
}

/// Calculate standard deviation
fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Calculate exponential moving average
fn ema(data: &[f64], period: usize) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema_val = data[0];

    for &val in data.iter().skip(1) {
        ema_val = alpha * val + (1.0 - alpha) * ema_val;
    }

    ema_val
}

/// Generate synthetic candles for testing
pub fn generate_synthetic_candles(n: usize, initial_price: f64) -> Vec<Candle> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.02).unwrap();
    let volume_normal = Normal::new(1000.0, 200.0).unwrap();

    let mut candles = Vec::with_capacity(n);
    let mut price = initial_price;

    for i in 0..n {
        let ret = normal.sample(&mut rng);
        let open = price;
        let close = open * (1.0 + ret);

        let high_extra = rng.gen::<f64>() * 0.01;
        let low_extra = rng.gen::<f64>() * 0.01;

        let high = open.max(close) * (1.0 + high_extra);
        let low = open.min(close) * (1.0 - low_extra);

        let volume_sample: f64 = volume_normal.sample(&mut rng);
        let volume = volume_sample.max(100.0);

        let timestamp = Utc::now() + chrono::Duration::hours(i as i64);

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_features() {
        let candles = generate_synthetic_candles(50, 100.0);
        let features = compute_market_features(&candles, 20);

        assert_eq!(features.len(), 9);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_normalize_features() {
        let candles = generate_synthetic_candles(100, 100.0);
        let features = compute_features_batch(&candles, 20);

        let (normalized, means, stds) = normalize_features(&features);

        assert_eq!(normalized.shape(), features.shape());
        assert_eq!(means.len(), 9);
        assert_eq!(stds.len(), 9);

        // Check normalization worked (mean should be ~0, std ~1)
        for j in 0..9 {
            let col = normalized.column(j);
            let mean = col.mean().unwrap();
            let std = std_dev(col.as_slice().unwrap());

            assert!((mean).abs() < 0.1, "Mean should be close to 0");
            assert!((std - 1.0).abs() < 0.1, "Std should be close to 1");
        }
    }
}
