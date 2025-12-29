//! Feature engineering for regime detection
//!
//! Provides technical indicators and features for HMM regime detection.

use super::types::{Candle, Dataset};
use ndarray::{Array1, Array2, Axis};
use statrs::statistics::Statistics;

/// Feature matrix with named columns
#[derive(Debug, Clone)]
pub struct Features {
    /// Feature matrix (rows = observations, cols = features)
    pub data: Array2<f64>,
    /// Feature names
    pub names: Vec<String>,
    /// Timestamps corresponding to each row
    pub timestamps: Vec<u64>,
}

impl Features {
    /// Number of observations
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Number of features
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<Array1<f64>> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|idx| self.data.column(idx).to_owned())
    }

    /// Standardize features (z-score normalization)
    pub fn standardize(&self) -> Features {
        let mut data = self.data.clone();

        for mut col in data.columns_mut() {
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(1.0);

            if std > 1e-10 {
                col.mapv_inplace(|x| (x - mean) / std);
            } else {
                col.mapv_inplace(|x| x - mean);
            }
        }

        Features {
            data,
            names: self.names.clone(),
            timestamps: self.timestamps.clone(),
        }
    }

    /// Get a slice of features
    pub fn slice(&self, start: usize, end: usize) -> Features {
        Features {
            data: self.data.slice(ndarray::s![start..end, ..]).to_owned(),
            names: self.names.clone(),
            timestamps: self.timestamps[start..end].to_vec(),
        }
    }
}

/// Feature builder for constructing feature matrix
pub struct FeatureBuilder {
    /// Window size for rolling calculations
    pub return_window: usize,
    /// Window size for volatility
    pub volatility_window: usize,
    /// RSI period
    pub rsi_period: usize,
    /// MACD parameters (fast, slow, signal)
    pub macd_params: (usize, usize, usize),
    /// Whether to standardize output
    pub standardize: bool,
}

impl Default for FeatureBuilder {
    fn default() -> Self {
        Self {
            return_window: 20,
            volatility_window: 20,
            rsi_period: 14,
            macd_params: (12, 26, 9),
            standardize: true,
        }
    }
}

impl FeatureBuilder {
    /// Create new feature builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set return window
    pub fn with_return_window(mut self, window: usize) -> Self {
        self.return_window = window;
        self
    }

    /// Set volatility window
    pub fn with_volatility_window(mut self, window: usize) -> Self {
        self.volatility_window = window;
        self
    }

    /// Set RSI period
    pub fn with_rsi_period(mut self, period: usize) -> Self {
        self.rsi_period = period;
        self
    }

    /// Build features from dataset
    pub fn build(&self, dataset: &Dataset) -> anyhow::Result<Features> {
        let n = dataset.len();
        let min_window = self
            .return_window
            .max(self.volatility_window)
            .max(self.macd_params.1);

        if n < min_window + 10 {
            anyhow::bail!(
                "Not enough data: need at least {} candles, got {}",
                min_window + 10,
                n
            );
        }

        let closes: Vec<f64> = dataset.closes();
        let volumes: Vec<f64> = dataset.volumes();
        let returns = dataset.log_returns();
        let timestamps = dataset.timestamps();

        // Calculate features
        let return_n = rolling_return(&closes, self.return_window);
        let volatility = rolling_volatility(&returns, self.volatility_window);
        let rsi = calculate_rsi(&closes, self.rsi_period);
        let (macd, signal) = calculate_macd(
            &closes,
            self.macd_params.0,
            self.macd_params.1,
            self.macd_params.2,
        );
        let volume_ratio = volume_ratio(&volumes, self.return_window);
        let momentum = momentum(&closes, 10);
        let price_range = price_range(&dataset.candles, self.return_window);

        // Align all features (they have different start offsets)
        let start_idx = min_window;
        let valid_len = n - start_idx - 1; // -1 because returns has one less element

        let mut feature_data = Vec::new();
        let feature_names = vec![
            format!("return_{}", self.return_window),
            format!("volatility_{}", self.volatility_window),
            format!("rsi_{}", self.rsi_period),
            "macd".to_string(),
            "macd_signal".to_string(),
            "macd_histogram".to_string(),
            "volume_ratio".to_string(),
            "momentum_10".to_string(),
            format!("price_range_{}", self.return_window),
        ];

        for i in 0..valid_len {
            let idx = start_idx + i;
            let row = vec![
                return_n.get(idx).copied().unwrap_or(0.0),
                volatility.get(idx - 1).copied().unwrap_or(0.0), // returns is 1 shorter
                rsi.get(idx).copied().unwrap_or(50.0),
                macd.get(idx).copied().unwrap_or(0.0),
                signal.get(idx).copied().unwrap_or(0.0),
                macd.get(idx).copied().unwrap_or(0.0) - signal.get(idx).copied().unwrap_or(0.0),
                volume_ratio.get(idx).copied().unwrap_or(1.0),
                momentum.get(idx).copied().unwrap_or(0.0),
                price_range.get(idx).copied().unwrap_or(0.0),
            ];
            feature_data.extend(row);
        }

        let data =
            Array2::from_shape_vec((valid_len, feature_names.len()), feature_data)?;
        let valid_timestamps = timestamps[start_idx + 1..start_idx + 1 + valid_len].to_vec();

        let features = Features {
            data,
            names: feature_names,
            timestamps: valid_timestamps,
        };

        if self.standardize {
            Ok(features.standardize())
        } else {
            Ok(features)
        }
    }
}

/// Convenience function to build features with default settings
pub fn build_features(candles: &[Candle]) -> anyhow::Result<Features> {
    let dataset = Dataset::new(candles.to_vec(), "UNKNOWN", "unknown");
    FeatureBuilder::default().build(&dataset)
}

// Helper functions for technical indicators

fn rolling_return(prices: &[f64], window: usize) -> Vec<f64> {
    prices
        .windows(window + 1)
        .map(|w| (w[window] - w[0]) / w[0])
        .collect()
}

fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![];
    }

    returns
        .windows(window)
        .map(|w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            variance.sqrt()
        })
        .collect()
}

fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![];
    }

    let mut rsi_values = vec![50.0; period]; // Fill initial values

    let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    for i in period..changes.len() {
        let window = &changes[i - period..i];
        let gains: f64 = window.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = window.iter().filter(|&&x| x < 0.0).map(|x| x.abs()).sum();

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        let rs = if avg_loss > 1e-10 {
            avg_gain / avg_loss
        } else {
            100.0
        };

        rsi_values.push(100.0 - 100.0 / (1.0 + rs));
    }

    rsi_values
}

fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() || period == 0 {
        return vec![];
    }

    let k = 2.0 / (period as f64 + 1.0);
    let mut ema = vec![prices[0]];

    for i in 1..prices.len() {
        let prev = ema[i - 1];
        ema.push(prices[i] * k + prev * (1.0 - k));
    }

    ema
}

fn calculate_macd(
    prices: &[f64],
    fast: usize,
    slow: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let ema_fast = calculate_ema(prices, fast);
    let ema_slow = calculate_ema(prices, slow);

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = calculate_ema(&macd_line, signal_period);

    (macd_line, signal_line)
}

fn volume_ratio(volumes: &[f64], window: usize) -> Vec<f64> {
    if volumes.len() < window {
        return vec![];
    }

    let mut result = vec![1.0; window - 1];

    for i in window - 1..volumes.len() {
        let window_slice = &volumes[i + 1 - window..i + 1];
        let avg = window_slice.iter().sum::<f64>() / window as f64;
        if avg > 1e-10 {
            result.push(volumes[i] / avg);
        } else {
            result.push(1.0);
        }
    }

    result
}

fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![];
    }

    let mut result = vec![0.0; period];

    for i in period..prices.len() {
        result.push((prices[i] - prices[i - period]) / prices[i - period]);
    }

    result
}

fn price_range(candles: &[Candle], window: usize) -> Vec<f64> {
    if candles.len() < window {
        return vec![];
    }

    let mut result = vec![0.0; window - 1];

    for i in window - 1..candles.len() {
        let window_slice = &candles[i + 1 - window..i + 1];
        let high = window_slice.iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let low = window_slice.iter().map(|c| c.low).fold(f64::MAX, f64::min);
        let mid = (high + low) / 2.0;
        if mid > 1e-10 {
            result.push((high - low) / mid);
        } else {
            result.push(0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_return() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let returns = rolling_return(&prices, 2);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let rsi = calculate_rsi(&prices, 14);
        assert!(!rsi.is_empty());
        // Trending up, RSI should be high
        assert!(rsi.last().unwrap_or(&50.0) > &50.0);
    }

    #[test]
    fn test_ema() {
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let ema = calculate_ema(&prices, 3);
        assert_eq!(ema.len(), 5);
        assert!((ema[0] - 10.0).abs() < 1e-10);
    }
}
