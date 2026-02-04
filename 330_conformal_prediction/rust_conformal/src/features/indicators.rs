//! Technical indicators for feature engineering

use crate::api::types::Kline;

/// Technical indicator calculations
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate simple moving average
    pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..data.len() {
            let sum: f64 = data[i + 1 - period..=i].iter().sum();
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate exponential moving average
    pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![data[0]]; // Start with first value

        for i in 1..data.len() {
            let ema = (data[i] - result[i - 1]) * multiplier + result[i - 1];
            result.push(ema);
        }

        // Set initial values to NaN
        for i in 0..period.saturating_sub(1).min(result.len()) {
            result[i] = f64::NAN;
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period + 1 {
            return vec![f64::NAN; data.len()];
        }

        let mut gains = Vec::with_capacity(data.len());
        let mut losses = Vec::with_capacity(data.len());

        gains.push(0.0);
        losses.push(0.0);

        for i in 1..data.len() {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let avg_gains = Self::ema(&gains, period);
        let avg_losses = Self::ema(&losses, period);

        let mut result = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            if avg_gains[i].is_nan() || avg_losses[i].is_nan() {
                result.push(f64::NAN);
            } else if avg_losses[i] < 1e-10 {
                result.push(100.0);
            } else {
                let rs = avg_gains[i] / avg_losses[i];
                result.push(100.0 - 100.0 / (1.0 + rs));
            }
        }

        result
    }

    /// Calculate MACD
    pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(data, fast);
        let ema_slow = Self::ema(data, slow);

        let mut macd_line = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            if ema_fast[i].is_nan() || ema_slow[i].is_nan() {
                macd_line.push(f64::NAN);
            } else {
                macd_line.push(ema_fast[i] - ema_slow[i]);
            }
        }

        let signal_line = Self::ema(&macd_line, signal);

        let mut histogram = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            if macd_line[i].is_nan() || signal_line[i].is_nan() {
                histogram.push(f64::NAN);
            } else {
                histogram.push(macd_line[i] - signal_line[i]);
            }
        }

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(data, period);
        let std = Self::rolling_std(data, period);

        let mut upper = Vec::with_capacity(data.len());
        let mut lower = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            if sma[i].is_nan() || std[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                upper.push(sma[i] + std_dev * std[i]);
                lower.push(sma[i] - std_dev * std[i]);
            }
        }

        (upper, sma, lower)
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..data.len() {
            let window = &data[i + 1 - period..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result.push(variance.sqrt());
        }

        result
    }

    /// Calculate Average True Range (ATR)
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.is_empty() {
            return vec![];
        }

        let mut true_ranges = vec![klines[0].high - klines[0].low];

        for i in 1..klines.len() {
            let tr = klines[i].true_range(Some(klines[i - 1].close));
            true_ranges.push(tr);
        }

        Self::ema(&true_ranges, period)
    }

    /// Calculate returns
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

    /// Calculate log returns
    pub fn log_returns(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() <= period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..data.len() {
            if data[i] > 0.0 && data[i - period] > 0.0 {
                result.push((data[i] / data[i - period]).ln());
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }

    /// Calculate momentum
    pub fn momentum(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() <= period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..data.len() {
            result.push(data[i] - data[i - period]);
        }

        result
    }

    /// Calculate Stochastic Oscillator
    pub fn stochastic(klines: &[Kline], period: usize) -> (Vec<f64>, Vec<f64>) {
        if klines.len() < period {
            return (vec![f64::NAN; klines.len()], vec![f64::NAN; klines.len()]);
        }

        let mut k_values = vec![f64::NAN; period - 1];

        for i in (period - 1)..klines.len() {
            let window = &klines[i + 1 - period..=i];
            let highest: f64 = window.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max);
            let lowest: f64 = window.iter().map(|k| k.low).fold(f64::INFINITY, f64::min);

            if (highest - lowest).abs() < 1e-10 {
                k_values.push(50.0);
            } else {
                let k = 100.0 * (klines[i].close - lowest) / (highest - lowest);
                k_values.push(k);
            }
        }

        let d_values = Self::sma(&k_values, 3);

        (k_values, d_values)
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
        assert!((returns[2] - (-0.04545454545)).abs() < 1e-6);
    }

    #[test]
    fn test_rsi() {
        let data: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let rsi = TechnicalIndicators::rsi(&data, 14);

        // RSI should be between 0 and 100
        for val in rsi.iter().filter(|v| !v.is_nan()) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }
}
