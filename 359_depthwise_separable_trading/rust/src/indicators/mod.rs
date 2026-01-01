//! Technical indicators module
//!
//! Provides common technical analysis indicators for trading strategies.

mod technical;

pub use technical::TechnicalIndicators;

use ndarray::Array1;

/// Indicator calculation trait
pub trait Indicator {
    /// Calculate indicator values
    fn calculate(&self, data: &Array1<f64>) -> Array1<f64>;

    /// Get indicator name
    fn name(&self) -> &str;

    /// Get lookback period required
    fn lookback(&self) -> usize;
}

/// Moving average types
#[derive(Debug, Clone, Copy)]
pub enum MovingAverageType {
    /// Simple Moving Average
    SMA,
    /// Exponential Moving Average
    EMA,
    /// Weighted Moving Average
    WMA,
}

/// Calculate Simple Moving Average
pub fn sma(data: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    for i in period - 1..n {
        let sum: f64 = data.slice(ndarray::s![i + 1 - period..=i]).sum();
        result[i] = sum / period as f64;
    }

    result
}

/// Calculate Exponential Moving Average
pub fn ema(data: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    if n == 0 || period == 0 {
        return result;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Initialize with SMA
    if n >= period {
        let initial_sum: f64 = data.slice(ndarray::s![..period]).sum();
        result[period - 1] = initial_sum / period as f64;

        // Calculate EMA
        for i in period..n {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
    }

    result
}

/// Calculate Weighted Moving Average
pub fn wma(data: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    let weight_sum: f64 = (1..=period).map(|i| i as f64).sum();

    for i in period - 1..n {
        let mut weighted_sum = 0.0;
        for j in 0..period {
            weighted_sum += data[i - period + 1 + j] * (j + 1) as f64;
        }
        result[i] = weighted_sum / weight_sum;
    }

    result
}

/// Calculate standard deviation
pub fn std_dev(data: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);
    let ma = sma(data, period);

    for i in period - 1..n {
        let slice = data.slice(ndarray::s![i + 1 - period..=i]);
        let mean = ma[i];
        let variance: f64 = slice.mapv(|x| (x - mean).powi(2)).sum() / period as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Calculate rate of change
pub fn roc(data: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    for i in period..n {
        if data[i - period] != 0.0 {
            result[i] = (data[i] - data[i - period]) / data[i - period] * 100.0;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let result = sma(&data, 3);

        assert!((result[2] - 2.0).abs() < 1e-10); // (1+2+3)/3 = 2
        assert!((result[9] - 9.0).abs() < 1e-10); // (8+9+10)/3 = 9
    }

    #[test]
    fn test_ema() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ema(&data, 3);

        // EMA should be calculated after initial SMA
        assert!(result[2] > 0.0);
        assert!(result[4] > 0.0);
    }

    #[test]
    fn test_roc() {
        let data = Array1::from_vec(vec![100.0, 110.0, 121.0, 133.1]);
        let result = roc(&data, 1);

        assert!((result[1] - 10.0).abs() < 1e-10);
        assert!((result[2] - 10.0).abs() < 1e-10);
    }
}
