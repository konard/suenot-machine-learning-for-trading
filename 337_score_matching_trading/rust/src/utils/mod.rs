//! # Utilities Module
//!
//! Common utilities for data processing and feature engineering.

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// OHLCV Candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Candle timestamp
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

impl Candle {
    /// Calculate return from previous candle
    pub fn return_from(&self, prev: &Candle) -> f64 {
        (self.close - prev.close) / prev.close
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Calculate typical price
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Market state representation
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Multi-scale returns
    pub returns: Vec<f64>,
    /// Volatility
    pub volatility: f64,
    /// Volatility ratio (short-term / long-term)
    pub volatility_ratio: f64,
    /// Momentum
    pub momentum: f64,
    /// Volume ratio
    pub volume_ratio: f64,
    /// Price position in recent range
    pub price_position: f64,
    /// RSI-like indicator
    pub rsi: f64,
    /// ATR ratio
    pub atr_ratio: f64,
}

impl MarketState {
    /// Convert to feature vector
    pub fn to_array(&self) -> Array1<f64> {
        let mut features = self.returns.clone();
        features.push(self.volatility);
        features.push(self.volatility_ratio);
        features.push(self.momentum);
        features.push(self.volume_ratio);
        features.push(self.price_position);
        features.push(self.rsi);
        features.push(self.atr_ratio);
        Array1::from_vec(features)
    }
}

/// Compute market state from candle data
///
/// # Arguments
///
/// * `candles` - Slice of candles (newest last)
/// * `lookback` - Lookback period for calculations
///
/// # Returns
///
/// Feature vector for score matching
pub fn compute_market_state(candles: &[Candle], lookback: usize) -> Array1<f64> {
    let n = candles.len();
    if n < lookback + 1 {
        return Array1::zeros(10);
    }

    // Calculate returns
    let returns: Vec<f64> = candles.windows(2)
        .map(|w| w[1].return_from(&w[0]))
        .collect();

    let recent_returns = &returns[returns.len().saturating_sub(lookback)..];

    // Multi-scale return sums
    let return_1 = *returns.last().unwrap_or(&0.0);
    let return_5 = recent_returns.iter().rev().take(5).sum::<f64>();
    let return_10 = recent_returns.iter().rev().take(10).sum::<f64>();
    let return_20 = recent_returns.iter().sum::<f64>();

    // Volatility (standard deviation of returns)
    let mean_return = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
    let volatility = {
        let variance = recent_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / recent_returns.len() as f64;
        variance.sqrt()
    };

    // Short-term vs long-term volatility
    let short_returns = &recent_returns[recent_returns.len().saturating_sub(5)..];
    let short_vol = {
        let mean = short_returns.iter().sum::<f64>() / short_returns.len().max(1) as f64;
        let var = short_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / short_returns.len().max(1) as f64;
        var.sqrt()
    };
    let volatility_ratio = if volatility > 0.0 { short_vol / volatility } else { 1.0 };

    // Momentum
    let first_price = candles[n.saturating_sub(lookback)].close;
    let last_price = candles[n - 1].close;
    let momentum = (last_price - first_price) / first_price;

    // Volume ratio
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let recent_volumes = &volumes[volumes.len().saturating_sub(lookback)..];
    let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
    let volume_ratio = if avg_volume > 0.0 {
        candles.last().map(|c| c.volume / avg_volume).unwrap_or(1.0)
    } else {
        1.0
    };

    // Price position in range
    let recent_candles = &candles[n.saturating_sub(lookback)..];
    let high = recent_candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let low = recent_candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let range = high - low;
    let price_position = if range > 0.0 {
        (last_price - low) / range
    } else {
        0.5
    };

    // RSI-like feature
    let gains: f64 = recent_returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = recent_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let rsi = if gains + losses > 0.0 { gains / (gains + losses) } else { 0.5 };

    // ATR ratio
    let trs: Vec<f64> = candles.windows(2)
        .map(|w| w[1].true_range(w[0].close))
        .collect();
    let recent_trs = &trs[trs.len().saturating_sub(lookback)..];
    let avg_tr = recent_trs.iter().sum::<f64>() / recent_trs.len().max(1) as f64;
    let current_tr = trs.last().copied().unwrap_or(0.0);
    let atr_ratio = if avg_tr > 0.0 { current_tr / avg_tr } else { 1.0 };

    Array1::from_vec(vec![
        return_1,
        return_5,
        return_10,
        return_20,
        volatility,
        volatility_ratio,
        momentum,
        volume_ratio,
        price_position,
        rsi,
        // atr_ratio, // Commented out to keep 10 features
    ])
}

/// Normalize features using z-score normalization
///
/// # Arguments
///
/// * `features` - Feature matrix (samples x features)
///
/// # Returns
///
/// Tuple of (normalized features, means, stds)
pub fn normalize_features(features: &ndarray::Array2<f64>) -> (ndarray::Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_features = features.ncols();

    let mut means = Array1::zeros(n_features);
    let mut stds = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = features.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let std = {
            let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
            variance.sqrt().max(1e-8)
        };
        means[j] = mean;
        stds[j] = std;
    }

    let normalized = features.mapv(|x| x) - &means + &stds.mapv(|_| 0.0); // placeholder
    let mut normalized = features.clone();

    for j in 0..n_features {
        for i in 0..features.nrows() {
            normalized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
        }
    }

    (normalized, means, stds)
}

/// Rolling statistics calculator
pub struct RollingStats {
    /// Window size
    window: usize,
    /// Values in window
    values: Vec<f64>,
    /// Current sum
    sum: f64,
    /// Current sum of squares
    sum_sq: f64,
}

impl RollingStats {
    /// Create new rolling stats calculator
    pub fn new(window: usize) -> Self {
        Self {
            window,
            values: Vec::with_capacity(window),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Add new value
    pub fn push(&mut self, value: f64) {
        if self.values.len() >= self.window {
            let old = self.values.remove(0);
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        self.values.push(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get current standard deviation
    pub fn std(&self) -> f64 {
        if self.values.len() < 2 {
            0.0
        } else {
            let n = self.values.len() as f64;
            let variance = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
            variance.max(0.0).sqrt()
        }
    }

    /// Check if window is full
    pub fn is_full(&self) -> bool {
        self.values.len() >= self.window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_candle(close: f64) -> Candle {
        Candle {
            timestamp: Utc::now(),
            open: close - 1.0,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_candle_return() {
        let c1 = create_test_candle(100.0);
        let c2 = create_test_candle(105.0);

        let ret = c2.return_from(&c1);
        assert!((ret - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_compute_market_state() {
        let candles: Vec<Candle> = (0..30)
            .map(|i| create_test_candle(100.0 + i as f64))
            .collect();

        let state = compute_market_state(&candles, 20);
        assert_eq!(state.len(), 10);
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(3);

        stats.push(1.0);
        stats.push(2.0);
        stats.push(3.0);

        assert_eq!(stats.mean(), 2.0);
        assert!(!stats.std().is_nan());

        stats.push(4.0);
        assert_eq!(stats.mean(), 3.0);
    }
}
