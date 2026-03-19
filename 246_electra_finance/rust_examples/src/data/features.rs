//! Feature engineering for ELECTRA-based trading
//!
//! This module provides functions for computing technical indicators
//! and constructing feature vectors from market data.

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering for market data
pub struct FeatureEngineering {
    sma_periods: Vec<usize>,
    ema_periods: Vec<usize>,
}

impl FeatureEngineering {
    /// Create a new feature engineering instance
    pub fn new() -> Self {
        Self {
            sma_periods: vec![5, 10, 20, 50],
            ema_periods: vec![12, 26],
        }
    }

    /// Compute all features from kline data
    ///
    /// Returns a 2D array of shape (num_samples, num_features)
    pub fn compute_features(&self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let returns = self.compute_returns(&closes);
        let log_returns = self.compute_log_returns(&closes);
        let volatility = self.rolling_std(&returns, 20);
        let rsi = self.compute_rsi(&closes, 14);
        let macd = self.compute_macd(&closes);
        let atr = self.compute_atr(&highs, &lows, &closes, 14);
        let volume_ratio = self.compute_volume_ratio(&volumes, 20);

        // Build feature matrix
        let num_features = 7;
        let mut features = Array2::zeros((n, num_features));

        for i in 0..n {
            features[[i, 0]] = returns[i];
            features[[i, 1]] = log_returns[i];
            features[[i, 2]] = volatility[i];
            features[[i, 3]] = rsi[i];
            features[[i, 4]] = macd[i];
            features[[i, 5]] = atr[i];
            features[[i, 6]] = volume_ratio[i];
        }

        features
    }

    /// Compute simple returns
    fn compute_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] != 0.0 {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Compute log returns
    fn compute_log_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 && prices[i] > 0.0 {
                returns.push((prices[i] / prices[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Compute rolling standard deviation
    fn rolling_std(&self, values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![0.0; n];

        for i in window..n {
            let slice = &values[i - window..i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let variance = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Compute RSI (Relative Strength Index)
    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![50.0; n];

        if n < period + 1 {
            return rsi;
        }

        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        rsi
    }

    /// Compute MACD (difference between 12-period and 26-period EMA)
    fn compute_macd(&self, prices: &[f64]) -> Vec<f64> {
        let ema12 = self.compute_ema(prices, 12);
        let ema26 = self.compute_ema(prices, 26);

        ema12
            .iter()
            .zip(ema26.iter())
            .map(|(e12, e26)| e12 - e26)
            .collect()
    }

    /// Compute Exponential Moving Average
    fn compute_ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut ema = vec![0.0; n];

        if n == 0 || period == 0 {
            return ema;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        ema[0] = values[0];

        for i in 1..n {
            ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }

        ema
    }

    /// Compute Average True Range
    fn compute_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        let n = highs.len();
        let mut atr = vec![0.0; n];

        if n < 2 {
            return atr;
        }

        let mut true_ranges = vec![0.0; n];
        true_ranges[0] = highs[0] - lows[0];

        for i in 1..n {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            true_ranges[i] = hl.max(hc).max(lc);
        }

        self.compute_ema(&true_ranges, period)
            .into_iter()
            .enumerate()
            .for_each(|(i, v)| atr[i] = v);

        atr
    }

    /// Compute volume ratio (current volume / average volume)
    fn compute_volume_ratio(&self, volumes: &[f64], period: usize) -> Vec<f64> {
        let n = volumes.len();
        let mut ratio = vec![1.0; n];

        for i in period..n {
            let avg = volumes[i - period..i].iter().sum::<f64>() / period as f64;
            if avg > 0.0 {
                ratio[i] = volumes[i] / avg;
            }
        }

        ratio
    }

    /// Normalize features to zero mean and unit variance
    pub fn normalize(&self, features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_features = features.ncols();
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);
        let mut normalized = features.clone();

        for j in 0..n_features {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);

            means[j] = mean;
            stds[j] = if std > 0.0 { std } else { 1.0 };

            for i in 0..features.nrows() {
                normalized[[i, j]] = (features[[i, j]] - mean) / stds[j];
            }
        }

        (normalized, means, stds)
    }
}

impl Default for FeatureEngineering {
    fn default() -> Self {
        Self::new()
    }
}
