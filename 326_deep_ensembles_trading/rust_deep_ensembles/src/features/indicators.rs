//! Technical indicators for feature engineering

use ndarray::Array1;

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate simple moving average
    pub fn sma(data: &[f64], period: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i >= period - 1 {
                let sum: f64 = data[i + 1 - period..=i].iter().sum();
                result[i] = sum / period as f64;
            } else {
                result[i] = data[i]; // Use current value for insufficient data
            }
        }

        result
    }

    /// Calculate exponential moving average
    pub fn ema(data: &[f64], period: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        let alpha = 2.0 / (period as f64 + 1.0);

        result[0] = data[0];
        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate standard deviation
    pub fn std(data: &[f64], period: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);
        let sma = Self::sma(data, period);

        for i in period - 1..n {
            let window = &data[i + 1 - period..=i];
            let mean = sma[i];
            let variance: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Calculate returns
    pub fn returns(data: &[f64]) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in 1..n {
            result[i] = (data[i] - data[i - 1]) / data[i - 1];
        }

        result
    }

    /// Calculate log returns
    pub fn log_returns(data: &[f64]) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in 1..n {
            result[i] = (data[i] / data[i - 1]).ln();
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(data: &[f64], period: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        if n < period + 1 {
            return result;
        }

        // Calculate price changes
        let changes: Vec<f64> = (1..n).map(|i| data[i] - data[i - 1]).collect();

        // Separate gains and losses
        let gains: Vec<f64> = changes.iter().map(|&c| c.max(0.0)).collect();
        let losses: Vec<f64> = changes.iter().map(|&c| (-c).max(0.0)).collect();

        // Calculate average gains and losses
        let avg_gains = Self::ema(&gains, period);
        let avg_losses = Self::ema(&losses, period);

        // Calculate RSI
        for i in 0..n - 1 {
            if avg_losses[i] > 0.0 {
                let rs = avg_gains[i] / avg_losses[i];
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                result[i + 1] = 100.0;
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let fast_ema = Self::ema(data, fast);
        let slow_ema = Self::ema(data, slow);

        let macd_line = &fast_ema - &slow_ema;
        let signal_line = Self::ema(macd_line.as_slice().unwrap(), signal);
        let histogram = &macd_line - &signal_line;

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(
        data: &[f64],
        period: usize,
        num_std: f64,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let middle = Self::sma(data, period);
        let std = Self::std(data, period);

        let upper = &middle + &std * num_std;
        let lower = &middle - &std * num_std;

        (upper, middle, lower)
    }

    /// Calculate Bollinger Band position (0 to 1)
    pub fn bb_position(data: &[f64], period: usize, num_std: f64) -> Array1<f64> {
        let (upper, _middle, lower) = Self::bollinger_bands(data, period, num_std);
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let range = upper[i] - lower[i];
            if range > 0.0 {
                result[i] = (data[i] - lower[i]) / range;
            } else {
                result[i] = 0.5;
            }
        }

        result
    }

    /// Calculate volatility (rolling standard deviation of returns)
    pub fn volatility(data: &[f64], period: usize) -> Array1<f64> {
        let returns = Self::log_returns(data);
        Self::std(returns.as_slice().unwrap(), period)
    }

    /// Calculate volume ratio (current / moving average)
    pub fn volume_ratio(volume: &[f64], period: usize) -> Array1<f64> {
        let ma = Self::sma(volume, period);
        let n = volume.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if ma[i] > 0.0 {
                result[i] = volume[i] / ma[i];
            } else {
                result[i] = 1.0;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::sma(&data, 3);

        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 105.0, 102.0];
        let returns = TechnicalIndicators::returns(&data);

        assert!((returns[1] - 0.05).abs() < 1e-10);
        assert!((returns[2] - (-0.02857142857)).abs() < 1e-6);
    }

    #[test]
    fn test_rsi() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0).collect();
        let rsi = TechnicalIndicators::rsi(&data, 14);

        // RSI should be between 0 and 100
        for i in 14..data.len() {
            assert!(rsi[i] >= 0.0 && rsi[i] <= 100.0);
        }
    }
}
