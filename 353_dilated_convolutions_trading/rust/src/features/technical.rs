//! Technical Indicators and Feature Extraction

use ndarray::{Array1, Array2};

use crate::api::Kline;

/// Technical features calculator
#[derive(Debug, Clone)]
pub struct TechnicalFeatures {
    /// Window sizes for moving averages
    ma_windows: Vec<usize>,
    /// Window size for volatility calculation
    volatility_window: usize,
}

impl Default for TechnicalFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalFeatures {
    /// Create a new feature calculator with default settings
    pub fn new() -> Self {
        Self {
            ma_windows: vec![5, 10, 20, 50],
            volatility_window: 20,
        }
    }

    /// Create with custom settings
    pub fn with_windows(ma_windows: Vec<usize>, volatility_window: usize) -> Self {
        Self {
            ma_windows,
            volatility_window,
        }
    }

    /// Calculate all features from klines
    ///
    /// Returns a feature matrix of shape (n_features, sequence_length)
    pub fn calculate(&self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        if n == 0 {
            return Array2::zeros((5, 0));
        }

        // Extract base series
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();

        // Calculate features
        let returns = self.calculate_returns(&closes);
        let log_volume = self.calculate_log_volume(&volumes);
        let range_pct = self.calculate_range_pct(&highs, &lows, &closes);
        let close_position = self.calculate_close_position(&highs, &lows, &closes);
        let volume_ma_ratio = self.calculate_volume_ma_ratio(&volumes, 20);

        // Stack features into matrix
        let mut features = Array2::zeros((5, n));
        for i in 0..n {
            features[[0, i]] = returns[i];
            features[[1, i]] = log_volume[i];
            features[[2, i]] = range_pct[i];
            features[[3, i]] = close_position[i];
            features[[4, i]] = volume_ma_ratio[i];
        }

        features
    }

    /// Calculate price returns
    pub fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Calculate log volume
    pub fn calculate_log_volume(&self, volumes: &[f64]) -> Vec<f64> {
        volumes.iter().map(|v| (v + 1.0).ln()).collect()
    }

    /// Calculate high-low range as percentage of close
    pub fn calculate_range_pct(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
        highs
            .iter()
            .zip(lows.iter())
            .zip(closes.iter())
            .map(|((h, l), c)| {
                if *c > 0.0 {
                    (h - l) / c
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate close position within bar (0 to 1)
    pub fn calculate_close_position(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
        highs
            .iter()
            .zip(lows.iter())
            .zip(closes.iter())
            .map(|((h, l), c)| {
                let range = h - l;
                if range > 0.0 {
                    (c - l) / range
                } else {
                    0.5
                }
            })
            .collect()
    }

    /// Calculate volume relative to its moving average
    pub fn calculate_volume_ma_ratio(&self, volumes: &[f64], window: usize) -> Vec<f64> {
        let ma = self.simple_moving_average(volumes, window);
        volumes
            .iter()
            .zip(ma.iter())
            .map(|(v, m)| if *m > 0.0 { v / m } else { 1.0 })
            .collect()
    }

    /// Calculate simple moving average
    pub fn simple_moving_average(&self, data: &[f64], window: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        let mut sum = 0.0;
        for i in 0..n {
            sum += data[i];
            if i >= window {
                sum -= data[i - window];
                result[i] = sum / window as f64;
            } else {
                result[i] = sum / (i + 1) as f64;
            }
        }

        result
    }

    /// Calculate exponential moving average
    pub fn exponential_moving_average(&self, data: &[f64], window: usize) -> Vec<f64> {
        let alpha = 2.0 / (window + 1) as f64;
        let mut ema = vec![data[0]];

        for i in 1..data.len() {
            let new_ema = alpha * data[i] + (1.0 - alpha) * ema[i - 1];
            ema.push(new_ema);
        }

        ema
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(&self, data: &[f64], window: usize) -> Vec<f64> {
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

    /// Calculate RSI (Relative Strength Index)
    pub fn calculate_rsi(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let returns = self.calculate_returns(prices);
        let n = returns.len();
        let mut rsi = vec![50.0; n];

        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        // Initial average
        for i in 1..window.min(n) {
            if returns[i] > 0.0 {
                avg_gain += returns[i];
            } else {
                avg_loss -= returns[i];
            }
        }
        avg_gain /= window as f64;
        avg_loss /= window as f64;

        // Calculate RSI
        for i in window..n {
            if returns[i] > 0.0 {
                avg_gain = (avg_gain * (window - 1) as f64 + returns[i]) / window as f64;
                avg_loss = avg_loss * (window - 1) as f64 / window as f64;
            } else {
                avg_gain = avg_gain * (window - 1) as f64 / window as f64;
                avg_loss = (avg_loss * (window - 1) as f64 - returns[i]) / window as f64;
            }

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - 100.0 / (1.0 + rs);
            } else {
                rsi[i] = 100.0;
            }
        }

        rsi
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn calculate_macd(&self, prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let fast_ema = self.exponential_moving_average(prices, fast);
        let slow_ema = self.exponential_moving_average(prices, slow);

        let macd_line: Vec<f64> = fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = self.exponential_moving_average(&macd_line, signal);

        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn calculate_bollinger_bands(&self, prices: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = self.simple_moving_average(prices, window);
        let std = self.rolling_std(prices, window);

        let upper: Vec<f64> = middle
            .iter()
            .zip(std.iter())
            .map(|(m, s)| m + num_std * s)
            .collect();

        let lower: Vec<f64> = middle
            .iter()
            .zip(std.iter())
            .map(|(m, s)| m - num_std * s)
            .collect();

        (upper, middle, lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_returns() {
        let features = TechnicalFeatures::new();
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let returns = features.calculate_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert_eq!(returns[0], 0.0);
        assert!((returns[1] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_sma() {
        let features = TechnicalFeatures::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = features.simple_moving_average(&data, 3);

        assert_eq!(sma.len(), 5);
        assert!((sma[2] - 2.0).abs() < 1e-10); // (1+2+3)/3
        assert!((sma[4] - 4.0).abs() < 1e-10); // (3+4+5)/3
    }

    #[test]
    fn test_rsi() {
        let features = TechnicalFeatures::new();
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let rsi = features.calculate_rsi(&prices, 14);

        // All gains, RSI should be close to 100
        assert!(rsi.last().unwrap() > &90.0);
    }
}
