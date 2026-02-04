//! Technical indicators implementation

use crate::api::types::Kline;

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate Simple Moving Average
    pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate Exponential Moving Average
    pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() || period == 0 {
            return vec![];
        }

        let mut result = vec![f64::NAN; data.len()];
        let multiplier = 2.0 / (period + 1) as f64;

        // First EMA is SMA
        if data.len() >= period {
            let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = sma;

            for i in period..data.len() {
                result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() <= period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; data.len()];
        let mut gains = Vec::with_capacity(data.len() - 1);
        let mut losses = Vec::with_capacity(data.len() - 1);

        // Calculate price changes
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

        // First RSI value
        let avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            result[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Subsequent values using smoothed averages
        let mut prev_avg_gain = avg_gain;
        let mut prev_avg_loss = avg_loss;

        for i in period..gains.len() {
            prev_avg_gain = (prev_avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            prev_avg_loss = (prev_avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if prev_avg_loss == 0.0 {
                result[i + 1] = 100.0;
            } else {
                let rs = prev_avg_gain / prev_avg_loss;
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(data, fast);
        let ema_slow = Self::ema(data, slow);

        // MACD line
        let macd_line: Vec<f64> = ema_fast.iter()
            .zip(ema_slow.iter())
            .map(|(&f, &s)| f - s)
            .collect();

        // Signal line (EMA of MACD)
        let signal_line = Self::ema(&macd_line, signal);

        // Histogram
        let histogram: Vec<f64> = macd_line.iter()
            .zip(signal_line.iter())
            .map(|(&m, &s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(data: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(data, period);
        let mut upper = vec![f64::NAN; data.len()];
        let mut lower = vec![f64::NAN; data.len()];

        for i in (period - 1)..data.len() {
            let slice = &data[(i + 1 - period)..=i];
            let mean = sma[i];
            let variance: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + num_std * std;
            lower[i] = mean - num_std * std;
        }

        (upper, sma, lower)
    }

    /// Calculate ATR (Average True Range)
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < 2 {
            return vec![f64::NAN; klines.len()];
        }

        let mut true_ranges = Vec::with_capacity(klines.len());
        true_ranges.push(klines[0].high - klines[0].low);

        for i in 1..klines.len() {
            let high_low = klines[i].high - klines[i].low;
            let high_close = (klines[i].high - klines[i - 1].close).abs();
            let low_close = (klines[i].low - klines[i - 1].close).abs();
            true_ranges.push(high_low.max(high_close).max(low_close));
        }

        Self::ema(&true_ranges, period)
    }

    /// Calculate returns
    pub fn returns(data: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0];
        for i in 1..data.len() {
            if data[i - 1] != 0.0 {
                result.push((data[i] - data[i - 1]) / data[i - 1]);
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// Calculate log returns
    pub fn log_returns(data: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0];
        for i in 1..data.len() {
            if data[i - 1] > 0.0 && data[i] > 0.0 {
                result.push((data[i] / data[i - 1]).ln());
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// Calculate rolling volatility
    pub fn volatility(data: &[f64], period: usize) -> Vec<f64> {
        let returns = Self::log_returns(data);
        let mut result = vec![f64::NAN; data.len()];

        for i in period..data.len() {
            let slice = &returns[(i + 1 - period)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
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

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 0.01);
        assert!((sma[3] - 3.0).abs() < 0.01);
        assert!((sma[4] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_rsi() {
        let data: Vec<f64> = (1..20).map(|i| 100.0 + i as f64).collect();
        let rsi = TechnicalIndicators::rsi(&data, 14);

        // All gains, RSI should be 100
        assert!((rsi[14] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 110.0, 99.0];
        let returns = TechnicalIndicators::returns(&data);

        assert!((returns[1] - 0.1).abs() < 0.01);
        assert!((returns[2] - (-0.1)).abs() < 0.01);
    }
}
