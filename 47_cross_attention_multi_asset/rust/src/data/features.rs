//! Feature engineering for financial time series
//!
//! Computes technical indicators and features for the model.

use super::Candle;
use serde::{Deserialize, Serialize};

/// Feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Window for moving average
    pub ma_window: usize,
    /// Window for volatility calculation
    pub volatility_window: usize,
    /// RSI period
    pub rsi_period: usize,
    /// MACD fast period
    pub macd_fast: usize,
    /// MACD slow period
    pub macd_slow: usize,
    /// Momentum period
    pub momentum_period: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            ma_window: 20,
            volatility_window: 20,
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            momentum_period: 5,
        }
    }
}

/// Computed features for each time step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Features {
    /// Log returns
    pub log_return: f64,
    /// Volume ratio (volume / SMA(volume))
    pub volume_ratio: f64,
    /// Historical volatility
    pub volatility: f64,
    /// RSI (Relative Strength Index)
    pub rsi: f64,
    /// MACD (Moving Average Convergence Divergence)
    pub macd: f64,
    /// Momentum
    pub momentum: f64,
}

impl Features {
    /// Convert to array
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.log_return,
            self.volume_ratio,
            self.volatility,
            self.rsi,
            self.macd,
            self.momentum,
        ]
    }

    /// Feature names
    pub fn names() -> [&'static str; 6] {
        [
            "log_return",
            "volume_ratio",
            "volatility",
            "rsi",
            "macd",
            "momentum",
        ]
    }
}

/// Compute features from candle data
pub fn compute_features(candles: &[Candle], config: &FeatureConfig) -> Vec<Features> {
    // Validate config to avoid panics and divide-by-zero
    if config.ma_window == 0
        || config.volatility_window == 0
        || config.rsi_period == 0
        || config.macd_fast == 0
        || config.macd_slow == 0
        || config.momentum_period == 0
    {
        return Vec::new();
    }

    let n = candles.len();
    if n < config.macd_slow + 1 {
        return Vec::new();
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // Compute log returns
    let log_returns: Vec<f64> = (1..n)
        .map(|i| (closes[i] / closes[i - 1]).ln())
        .collect();

    // Compute EMA for MACD
    let ema_fast = ema(&closes, config.macd_fast);
    let ema_slow = ema(&closes, config.macd_slow);

    // Compute volume SMA
    let volume_sma = sma(&volumes, config.ma_window);

    // Compute volatility
    let volatility = rolling_std(&log_returns, config.volatility_window);

    // Compute RSI
    let rsi = compute_rsi(&closes, config.rsi_period);

    // Compute momentum
    let momentum = compute_momentum(&closes, config.momentum_period);

    // Build features for each time step
    let start = config.macd_slow; // Start after we have enough data
    let mut features = Vec::with_capacity(n - start);

    for i in start..n {
        let log_ret = if i > 0 { log_returns[i - 1] } else { 0.0 };
        let vol_ratio = if volume_sma[i] > 0.0 {
            volumes[i] / volume_sma[i]
        } else {
            1.0
        };
        let vol = volatility.get(i - 1).copied().unwrap_or(0.0);
        let r = rsi.get(i).copied().unwrap_or(50.0);
        let macd_val = ema_fast.get(i).unwrap_or(&0.0) - ema_slow.get(i).unwrap_or(&0.0);
        let mom = momentum.get(i).copied().unwrap_or(0.0);
        let price = closes[i];

        features.push(Features {
            log_return: log_ret,
            volume_ratio: vol_ratio,
            volatility: vol,
            rsi: r / 100.0, // Normalize to [0, 1]
            macd: if price != 0.0 { macd_val / price } else { 0.0 }, // Normalize by price
            momentum: mom,
        });
    }

    features
}

/// Simple Moving Average
fn sma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let sum: f64 = data[start..=i].iter().sum();
        result[i] = sum / (i - start + 1) as f64;
    }

    result
}

/// Exponential Moving Average
fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = vec![0.0; n];
    let alpha = 2.0 / (period + 1) as f64;

    // Initialize with first value
    result[0] = data[0];

    for i in 1..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

/// Rolling standard deviation
fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let slice = &data[start..=i];
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Compute RSI (Relative Strength Index)
fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![50.0; n];

    if n < period + 1 {
        return result;
    }

    // Calculate price changes
    let changes: Vec<f64> = (1..n).map(|i| prices[i] - prices[i - 1]).collect();

    // Separate gains and losses
    let gains: Vec<f64> = changes.iter().map(|&c| if c > 0.0 { c } else { 0.0 }).collect();
    let losses: Vec<f64> = changes.iter().map(|&c| if c < 0.0 { -c } else { 0.0 }).collect();

    // Calculate EMA of gains and losses
    let avg_gain = ema(&gains, period);
    let avg_loss = ema(&losses, period);

    // Calculate RSI
    for i in period..n {
        let ag = avg_gain[i - 1];
        let al = avg_loss[i - 1];

        if al == 0.0 {
            result[i] = 100.0;
        } else if ag == 0.0 {
            result[i] = 0.0;
        } else {
            let rs = ag / al;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    result
}

/// Compute momentum
fn compute_momentum(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![0.0; n];

    for i in period..n {
        let denom = prices[i - period];
        result[i] = if denom != 0.0 {
            (prices[i] - denom) / denom
        } else {
            0.0
        };
    }

    result
}

/// Normalize features to zero mean and unit variance
pub fn normalize_features(features: &mut [Features], epsilon: f64) {
    if features.is_empty() {
        return;
    }

    let n = features.len() as f64;

    // Compute means
    let mut means = [0.0; 6];
    for f in features.iter() {
        let arr = f.to_array();
        for (i, val) in arr.iter().enumerate() {
            means[i] += val;
        }
    }
    for mean in &mut means {
        *mean /= n;
    }

    // Compute standard deviations
    let mut stds = [0.0; 6];
    for f in features.iter() {
        let arr = f.to_array();
        for (i, val) in arr.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n).sqrt() + epsilon;
    }

    // Normalize
    for f in features.iter_mut() {
        f.log_return = (f.log_return - means[0]) / stds[0];
        f.volume_ratio = (f.volume_ratio - means[1]) / stds[1];
        f.volatility = (f.volatility - means[2]) / stds[2];
        f.rsi = (f.rsi - means[3]) / stds[3];
        f.macd = (f.macd - means[4]) / stds[4];
        f.momentum = (f.momentum - means[5]) / stds[5];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle {
                    timestamp: i as i64 * 3600000,
                    open: base,
                    high: base + 1.0,
                    low: base - 1.0,
                    close: base + 0.5,
                    volume: 1000.0 + (i as f64 * 100.0),
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert!((result[4] - 4.0).abs() < 1e-10); // (3 + 4 + 5) / 3
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);

        assert_eq!(result[0], 1.0);
        assert!(result[4] > result[3]); // EMA should be increasing
    }

    #[test]
    fn test_compute_features() {
        let candles = create_test_candles(100);
        let config = FeatureConfig::default();
        let features = compute_features(&candles, &config);

        assert!(!features.is_empty());
        assert!(features.len() < candles.len()); // Some data lost at start
    }

    #[test]
    fn test_rsi_range() {
        let candles = create_test_candles(100);
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let rsi = compute_rsi(&prices, 14);

        for &r in &rsi {
            assert!(r >= 0.0 && r <= 100.0);
        }
    }

    #[test]
    fn test_normalize_features() {
        let candles = create_test_candles(100);
        let config = FeatureConfig::default();
        let mut features = compute_features(&candles, &config);

        normalize_features(&mut features, 1e-8);

        // Check that means are approximately zero
        let n = features.len() as f64;
        let mut means = [0.0; 6];
        for f in &features {
            let arr = f.to_array();
            for (i, val) in arr.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= n;
            assert!(mean.abs() < 1e-6, "Mean not zero: {}", mean);
        }
    }
}
