//! Technical indicators for feature engineering

use ndarray::Array1;

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate simple moving average
    pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..data.len() {
            let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate exponential moving average
    pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }

        let mut result = Vec::with_capacity(data.len());
        let alpha = 2.0 / (period as f64 + 1.0);

        result.push(data[0]);

        for i in 1..data.len() {
            let ema = alpha * data[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema);
        }

        result
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..data.len() {
            let window = &data[(i + 1 - period)..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result.push(variance.sqrt());
        }

        result
    }

    /// Calculate returns (percentage change)
    pub fn returns(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() <= period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..data.len() {
            if data[i - period] != 0.0 {
                result.push((data[i] - data[i - period]) / data[i - period]);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period + 1 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period];

        // Calculate price changes
        let changes: Vec<f64> = (1..data.len())
            .map(|i| data[i] - data[i - 1])
            .collect();

        // Calculate gains and losses
        let gains: Vec<f64> = changes.iter().map(|&c| if c > 0.0 { c } else { 0.0 }).collect();
        let losses: Vec<f64> = changes.iter().map(|&c| if c < 0.0 { -c } else { 0.0 }).collect();

        // First average
        let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        for i in period..changes.len() {
            // Smooth averages
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(data, fast);
        let ema_slow = Self::ema(data, slow);

        // MACD line = fast EMA - slow EMA
        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        // Signal line = EMA of MACD line
        let signal_line = Self::ema(&macd_line, signal);

        // Histogram = MACD - Signal
        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = Self::sma(data, period);
        let rolling_std = Self::rolling_std(data, period);

        let upper: Vec<f64> = middle
            .iter()
            .zip(rolling_std.iter())
            .map(|(m, s)| m + std_dev * s)
            .collect();

        let lower: Vec<f64> = middle
            .iter()
            .zip(rolling_std.iter())
            .map(|(m, s)| m - std_dev * s)
            .collect();

        (upper, middle, lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::sma(&data, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 110.0, 105.0, 115.0];
        let returns = TechnicalIndicators::returns(&data, 1);

        assert!(returns[0].is_nan());
        assert!((returns[1] - 0.1).abs() < 1e-10);
        assert!((returns[2] - (-0.0454545454545)).abs() < 1e-5);
    }

    #[test]
    fn test_rsi_bounds() {
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let rsi = TechnicalIndicators::rsi(&data, 14);

        for value in rsi.iter().skip(14) {
            if !value.is_nan() {
                assert!(*value >= 0.0 && *value <= 100.0);
            }
        }
    }
}
