//! Feature engineering for financial data
//!
//! This module provides utilities for computing technical indicators
//! and creating features from OHLCV data.

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering utilities
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Compute Simple Moving Average (SMA)
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = prices[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..prices.len() {
            sum = sum - prices[i - period] + prices[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Compute Exponential Moving Average (EMA)
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![prices[0]];

        for &price in &prices[1..] {
            let ema = (price - result.last().unwrap()) * multiplier + result.last().unwrap();
            result.push(ema);
        }

        result
    }

    /// Compute Relative Strength Index (RSI)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![f64::NAN; prices.len()];
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let mut result = vec![f64::NAN; period];

        // First RSI uses simple average
        let avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        let rs = if avg_loss == 0.0 { 100.0 } else { avg_gain / avg_loss };
        result.push(100.0 - (100.0 / (1.0 + rs)));

        // Subsequent RSI uses smoothed average
        let mut prev_avg_gain = avg_gain;
        let mut prev_avg_loss = avg_loss;

        for i in period..gains.len() {
            prev_avg_gain = (prev_avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            prev_avg_loss = (prev_avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            let rs = if prev_avg_loss == 0.0 { 100.0 } else { prev_avg_gain / prev_avg_loss };
            result.push(100.0 - (100.0 / (1.0 + rs)));
        }

        result
    }

    /// Compute MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = Self::ema(&macd_line, signal);

        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Compute Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, period);
        let mut upper = vec![f64::NAN; prices.len()];
        let mut lower = vec![f64::NAN; prices.len()];

        for i in (period - 1)..prices.len() {
            let window = &prices[i + 1 - period..=i];
            let mean = sma[i];
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + num_std * std;
            lower[i] = mean - num_std * std;
        }

        (upper, sma, lower)
    }

    /// Compute ATR (Average True Range)
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < period + 1 {
            return vec![f64::NAN; klines.len()];
        }

        let mut tr = vec![klines[0].high - klines[0].low];

        for i in 1..klines.len() {
            let high_low = klines[i].high - klines[i].low;
            let high_close = (klines[i].high - klines[i - 1].close).abs();
            let low_close = (klines[i].low - klines[i - 1].close).abs();
            tr.push(high_low.max(high_close).max(low_close));
        }

        Self::ema(&tr, period)
    }

    /// Normalize a series to [0, 1] range
    pub fn normalize(values: &[f64]) -> Vec<f64> {
        let valid: Vec<f64> = values.iter().filter(|x| !x.is_nan()).copied().collect();
        if valid.is_empty() {
            return vec![f64::NAN; values.len()];
        }

        let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range == 0.0 {
            return vec![0.5; values.len()];
        }

        values.iter().map(|x| (x - min) / range).collect()
    }

    /// Standardize a series (z-score normalization)
    pub fn standardize(values: &[f64]) -> Vec<f64> {
        let valid: Vec<f64> = values.iter().filter(|x| !x.is_nan()).copied().collect();
        if valid.is_empty() {
            return vec![f64::NAN; values.len()];
        }

        let mean: f64 = valid.iter().sum::<f64>() / valid.len() as f64;
        let variance: f64 = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return vec![0.0; values.len()];
        }

        values.iter().map(|x| (x - mean) / std).collect()
    }

    /// Compute price returns
    pub fn returns(prices: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0];
        for i in 1..prices.len() {
            result.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }
        result
    }

    /// Compute log returns
    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0];
        for i in 1..prices.len() {
            result.push((prices[i] / prices[i - 1]).ln());
        }
        result
    }

    /// Create OHLCV feature matrix from klines
    pub fn create_ohlcv_features(klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let mut features = Array2::zeros((n, 5));

        for (i, kline) in klines.iter().enumerate() {
            features[[i, 0]] = kline.open;
            features[[i, 1]] = kline.high;
            features[[i, 2]] = kline.low;
            features[[i, 3]] = kline.close;
            features[[i, 4]] = kline.volume;
        }

        features
    }

    /// Create normalized feature matrix for CNN input
    pub fn create_normalized_features(klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        if n == 0 {
            return Array2::zeros((0, 5));
        }

        let mut features = Array2::zeros((n, 5));

        // Get first close price for normalization
        let price_scale = klines[0].close;

        // Get volume stats
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let vol_mean: f64 = volumes.iter().sum::<f64>() / n as f64;
        let vol_std: f64 = (volumes.iter().map(|v| (v - vol_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let vol_std = if vol_std == 0.0 { 1.0 } else { vol_std };

        for (i, kline) in klines.iter().enumerate() {
            // Normalize prices relative to first close
            features[[i, 0]] = kline.open / price_scale - 1.0;
            features[[i, 1]] = kline.high / price_scale - 1.0;
            features[[i, 2]] = kline.low / price_scale - 1.0;
            features[[i, 3]] = kline.close / price_scale - 1.0;

            // Standardize volume
            features[[i, 4]] = (kline.volume - vol_mean) / vol_std;
        }

        features
    }

    /// Create feature array for a single sequence (for CNN input)
    ///
    /// Returns array of shape (channels, sequence_length) = (5, len)
    pub fn create_sequence_features(klines: &[Kline]) -> Array2<f64> {
        let normalized = Self::create_normalized_features(klines);

        // Transpose to (channels, sequence_length)
        normalized.t().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = FeatureEngineering::sma(&prices, 3);
        assert_eq!(sma.len(), 5);
        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 102.0, 101.0, 105.0];
        let returns = FeatureEngineering::returns(&prices);
        assert_eq!(returns.len(), 4);
        assert!((returns[0] - 0.0).abs() < 1e-10);
        assert!((returns[1] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let values = vec![0.0, 50.0, 100.0];
        let normalized = FeatureEngineering::normalize(&values);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }
}
