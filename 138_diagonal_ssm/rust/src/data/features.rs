//! Feature engineering for financial time series.
//!
//! Computes technical indicators from raw kline data, producing a feature
//! matrix suitable for model input.

use anyhow::Result;
use log::debug;
use ndarray::Array2;

use crate::api::types::Kline;

/// Feature engineering engine that computes technical indicators from klines.
///
/// Currently computes: returns, volatility, RSI, MACD, volume_change.
#[derive(Debug, Clone)]
pub struct FeatureEngine {
    /// Lookback window for volatility calculation.
    pub volatility_window: usize,
    /// Lookback window for RSI calculation.
    pub rsi_window: usize,
    /// Short EMA period for MACD.
    pub macd_short: usize,
    /// Long EMA period for MACD.
    pub macd_long: usize,
    /// Signal line period for MACD.
    pub macd_signal: usize,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self {
            volatility_window: 20,
            rsi_window: 14,
            macd_short: 12,
            macd_long: 26,
            macd_signal: 9,
        }
    }
}

impl FeatureEngine {
    /// Number of features produced by this engine.
    pub const NUM_FEATURES: usize = 5;

    /// Compute all features from kline data.
    ///
    /// Returns a 2D array of shape (n_valid_samples, NUM_FEATURES) where
    /// n_valid_samples = klines.len() - warmup_period.
    ///
    /// Feature columns: [returns, volatility, rsi, macd, volume_change]
    pub fn compute_all(&self, klines: &[Kline]) -> Result<Array2<f64>> {
        let n = klines.len();
        let warmup = self.macd_long.max(self.volatility_window).max(self.rsi_window);

        if n <= warmup {
            anyhow::bail!(
                "Not enough klines for feature computation: have {}, need > {}",
                n,
                warmup
            );
        }

        let valid_len = n - warmup;
        let mut features = Array2::<f64>::zeros((valid_len, Self::NUM_FEATURES));

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Compute raw indicators over the full series
        let returns = self.compute_returns(&closes);
        let volatility = self.compute_volatility(&returns, self.volatility_window);
        let rsi = self.compute_rsi(&closes, self.rsi_window);
        let macd = self.compute_macd(&closes);
        let volume_change = self.compute_volume_change(&volumes);

        // Fill the feature matrix (skip warmup period)
        for i in 0..valid_len {
            let idx = warmup + i;
            features[[i, 0]] = returns.get(idx).copied().unwrap_or(0.0);
            features[[i, 1]] = volatility.get(idx).copied().unwrap_or(0.0);
            features[[i, 2]] = rsi.get(idx).copied().unwrap_or(50.0);
            features[[i, 3]] = macd.get(idx).copied().unwrap_or(0.0);
            features[[i, 4]] = volume_change.get(idx).copied().unwrap_or(0.0);
        }

        debug!("Computed {} features for {} samples", Self::NUM_FEATURES, valid_len);
        Ok(features)
    }

    /// Log returns: ln(close[t] / close[t-1])
    fn compute_returns(&self, closes: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0]; // first return is zero
        for i in 1..closes.len() {
            if closes[i - 1] > 0.0 {
                returns.push((closes[i] / closes[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Rolling standard deviation of returns.
    fn compute_volatility(&self, returns: &[f64], window: usize) -> Vec<f64> {
        let n = returns.len();
        let mut vol = vec![0.0; n];

        for i in window..n {
            let slice = &returns[(i - window)..i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 =
                slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            vol[i] = variance.sqrt();
        }
        vol
    }

    /// Relative Strength Index.
    fn compute_rsi(&self, closes: &[f64], window: usize) -> Vec<f64> {
        let n = closes.len();
        let mut rsi = vec![50.0; n]; // neutral default

        if n < window + 1 {
            return rsi;
        }

        // Calculate price changes
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];
        for i in 1..n {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Initial average gain/loss
        let mut avg_gain: f64 = gains[1..=window].iter().sum::<f64>() / window as f64;
        let mut avg_loss: f64 = losses[1..=window].iter().sum::<f64>() / window as f64;

        for i in window..n {
            if i > window {
                // Smoothed moving average
                avg_gain = (avg_gain * (window as f64 - 1.0) + gains[i]) / window as f64;
                avg_loss = (avg_loss * (window as f64 - 1.0) + losses[i]) / window as f64;
            }

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            } else if avg_gain > 0.0 {
                rsi[i] = 100.0;
            } else {
                rsi[i] = 50.0;
            }
        }
        rsi
    }

    /// MACD line (short EMA - long EMA).
    fn compute_macd(&self, closes: &[f64]) -> Vec<f64> {
        let short_ema = ema(closes, self.macd_short);
        let long_ema = ema(closes, self.macd_long);

        short_ema
            .iter()
            .zip(long_ema.iter())
            .map(|(s, l)| s - l)
            .collect()
    }

    /// Volume change: (volume[t] - volume[t-1]) / volume[t-1]
    fn compute_volume_change(&self, volumes: &[f64]) -> Vec<f64> {
        let mut changes = vec![0.0];
        for i in 1..volumes.len() {
            if volumes[i - 1] > 0.0 {
                changes.push((volumes[i] - volumes[i - 1]) / volumes[i - 1]);
            } else {
                changes.push(0.0);
            }
        }
        changes
    }
}

/// Compute exponential moving average.
fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    if n == 0 || period == 0 {
        return result;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // SMA for the first `period` values
    let sma: f64 = data[..period.min(n)].iter().sum::<f64>() / period.min(n) as f64;
    if period <= n {
        result[period - 1] = sma;
    }

    // EMA for subsequent values
    for i in period..n {
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
    }
    result
}
