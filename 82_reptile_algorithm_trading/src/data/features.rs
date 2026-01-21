//! Feature generation for trading data.
//!
//! This module provides technical indicators and feature engineering
//! utilities for preparing market data for the Reptile model.

use crate::data::bybit::Kline;

/// Feature generator for trading data
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    /// Lookback window for rolling calculations
    window: usize,
}

impl FeatureGenerator {
    /// Create a new feature generator
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Calculate returns from prices
    pub fn calculate_returns(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![];
        }

        prices
            .windows(period + 1)
            .map(|w| (w[period] - w[0]) / w[0])
            .collect()
    }

    /// Calculate simple moving average
    pub fn sma(values: &[f64], window: usize) -> Vec<f64> {
        if values.len() < window {
            return vec![];
        }

        values
            .windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Calculate exponential moving average
    pub fn ema(values: &[f64], window: usize) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }

        let alpha = 2.0 / (window as f64 + 1.0);
        let mut ema = Vec::with_capacity(values.len());

        ema.push(values[0]);
        for &value in values.iter().skip(1) {
            let prev = *ema.last().unwrap();
            ema.push(alpha * value + (1.0 - alpha) * prev);
        }

        ema
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64], window: usize) -> Vec<f64> {
        if values.len() < window {
            return vec![];
        }

        values
            .windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / window as f64;
                let variance = w.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / window as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(prices: &[f64], window: usize) -> Vec<f64> {
        if prices.len() < window + 1 {
            return vec![];
        }

        let changes: Vec<f64> = prices.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for &change in &changes {
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let avg_gains = Self::sma(&gains, window);
        let avg_losses = Self::sma(&losses, window);

        avg_gains.iter()
            .zip(avg_losses.iter())
            .map(|(&gain, &loss)| {
                if loss == 0.0 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + gain / loss))
                }
            })
            .collect()
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        // Align the EMAs
        let offset = slow - fast;
        let macd_line: Vec<f64> = ema_fast.iter()
            .skip(offset)
            .zip(ema_slow.iter())
            .map(|(&fast, &slow)| fast - slow)
            .collect();

        let signal_line = Self::ema(&macd_line, signal);

        (macd_line, signal_line)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, window);
        let std = Self::std_dev(prices, window);

        let upper: Vec<f64> = sma.iter()
            .zip(std.iter())
            .map(|(&m, &s)| m + num_std * s)
            .collect();

        let lower: Vec<f64> = sma.iter()
            .zip(std.iter())
            .map(|(&m, &s)| m - num_std * s)
            .collect();

        (upper, sma, lower)
    }

    /// Generate features from klines
    pub fn generate_features(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        if klines.len() < self.window + 10 {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let _volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate various indicators
        let returns_1 = Self::calculate_returns(&closes, 1);
        let returns_5 = Self::calculate_returns(&closes, 5);
        let returns_10 = Self::calculate_returns(&closes, 10);

        let sma = Self::sma(&closes, self.window);
        let ema = Self::ema(&closes, self.window);

        let volatility = Self::std_dev(&returns_1, self.window);
        let rsi = Self::rsi(&closes, 14);

        // Align all features to the same length
        let min_len = returns_1.len()
            .min(returns_5.len())
            .min(returns_10.len())
            .min(sma.len())
            .min(ema.len())
            .min(volatility.len())
            .min(rsi.len());

        if min_len == 0 {
            return vec![];
        }

        // Take the last min_len values from each feature
        let offset_r1 = returns_1.len() - min_len;
        let offset_r5 = returns_5.len() - min_len;
        let offset_r10 = returns_10.len() - min_len;
        let offset_sma = sma.len() - min_len;
        let offset_ema = ema.len() - min_len;
        let offset_vol = volatility.len() - min_len;
        let offset_rsi = rsi.len() - min_len;
        let offset_close = closes.len() - min_len;

        let mut features = Vec::with_capacity(min_len);

        for i in 0..min_len {
            let close = closes[offset_close + i];
            let feature_vec = vec![
                returns_1[offset_r1 + i],
                returns_5[offset_r5 + i],
                returns_10[offset_r10 + i],
                close / sma[offset_sma + i] - 1.0,  // SMA ratio
                close / ema[offset_ema + i] - 1.0,  // EMA ratio
                volatility[offset_vol + i],
                rsi[offset_rsi + i] / 100.0 - 0.5,  // Normalized RSI
                (klines[offset_close + i].high - klines[offset_close + i].low) / close,  // Range ratio
            ];
            features.push(feature_vec);
        }

        features
    }

    /// Generate target labels (future returns)
    pub fn generate_targets(&self, klines: &[Kline], horizon: usize) -> Vec<f64> {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        Self::calculate_returns(&closes, horizon)
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<&'static str> {
        vec![
            "return_1d",
            "return_5d",
            "return_10d",
            "sma_ratio",
            "ema_ratio",
            "volatility",
            "rsi_normalized",
            "range_ratio",
        ]
    }

    /// Normalize features to zero mean and unit variance
    pub fn normalize_features(features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        if features.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let num_features = features[0].len();
        let num_samples = features.len();

        // Calculate means
        let mut means = vec![0.0; num_features];
        for sample in features {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= num_samples as f64;
        }

        // Calculate standard deviations
        let mut stds = vec![0.0; num_features];
        for sample in features {
            for (i, &val) in sample.iter().enumerate() {
                stds[i] += (val - means[i]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / num_samples as f64).sqrt().max(1e-8);
        }

        // Normalize
        let normalized: Vec<Vec<f64>> = features
            .iter()
            .map(|sample| {
                sample.iter()
                    .enumerate()
                    .map(|(i, &val)| (val - means[i]) / stds[i])
                    .collect()
            })
            .collect();

        (normalized, means, stds)
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = FeatureGenerator::sma(&values, 3);
        assert_eq!(sma.len(), 3);
        assert!((sma[0] - 2.0).abs() < 1e-10);
        assert!((sma[1] - 3.0).abs() < 1e-10);
        assert!((sma[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 105.0, 110.0, 100.0];
        let returns = FeatureGenerator::calculate_returns(&prices, 1);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let (normalized, means, _stds) = FeatureGenerator::normalize_features(&features);

        assert_eq!(normalized.len(), 3);
        assert!((means[0] - 3.0).abs() < 1e-10);
        assert!((means[1] - 4.0).abs() < 1e-10);

        // Check that normalized data has approximately zero mean
        let norm_mean: f64 = normalized.iter()
            .map(|v| v[0])
            .sum::<f64>() / normalized.len() as f64;
        assert!(norm_mean.abs() < 1e-10);
    }
}
