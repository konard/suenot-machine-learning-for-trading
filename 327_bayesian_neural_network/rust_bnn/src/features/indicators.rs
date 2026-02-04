//! Technical indicators for trading features.

use crate::api::types::Kline;

/// Technical indicators calculator.
#[derive(Debug, Clone, Default)]
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate Simple Moving Average.
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

    /// Calculate Exponential Moving Average.
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let mut result = vec![f64::NAN; prices.len()];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // Start with SMA
        if prices.len() >= period {
            let sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = sma;

            for i in period..prices.len() {
                result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }

        result
    }

    /// Calculate Relative Strength Index.
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; prices.len()];
        let mut gains = Vec::with_capacity(prices.len());
        let mut losses = Vec::with_capacity(prices.len());

        gains.push(0.0);
        losses.push(0.0);

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

        // Initial averages
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        if avg_loss > 0.0 {
            result[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        } else {
            result[period] = 100.0;
        }

        // Smoothed averages
        for i in (period + 1)..prices.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if avg_loss > 0.0 {
                result[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
            } else {
                result[i] = 100.0;
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence).
    pub fn macd(
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let fast_ema = Self::ema(prices, fast_period);
        let slow_ema = Self::ema(prices, slow_period);

        let macd_line: Vec<f64> = fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(&f, &s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        let valid_macd: Vec<f64> = macd_line.iter().filter(|v| !v.is_nan()).copied().collect();
        let signal_line_valid = Self::ema(&valid_macd, signal_period);

        // Map back to original length
        let mut signal_line = vec![f64::NAN; prices.len()];
        let mut valid_idx = 0;
        for i in 0..prices.len() {
            if !macd_line[i].is_nan() {
                if valid_idx < signal_line_valid.len() {
                    signal_line[i] = signal_line_valid[valid_idx];
                }
                valid_idx += 1;
            }
        }

        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(&m, &s)| {
                if m.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    m - s
                }
            })
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands.
    pub fn bollinger_bands(
        prices: &[f64],
        period: usize,
        std_dev: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = Self::sma(prices, period);
        let mut upper = vec![f64::NAN; prices.len()];
        let mut lower = vec![f64::NAN; prices.len()];

        for i in (period - 1)..prices.len() {
            let slice = &prices[(i + 1 - period)..=i];
            let mean = middle[i];
            let variance: f64 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }

        (upper, middle, lower)
    }

    /// Calculate Average True Range.
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < 2 {
            return vec![f64::NAN; klines.len()];
        }

        let mut tr = vec![f64::NAN; klines.len()];
        tr[0] = klines[0].high - klines[0].low;

        for i in 1..klines.len() {
            let high_low = klines[i].high - klines[i].low;
            let high_close = (klines[i].high - klines[i - 1].close).abs();
            let low_close = (klines[i].low - klines[i - 1].close).abs();
            tr[i] = high_low.max(high_close).max(low_close);
        }

        // ATR is SMA of True Range
        Self::sma(&tr, period)
    }

    /// Calculate Stochastic Oscillator.
    pub fn stochastic(klines: &[Kline], k_period: usize, d_period: usize) -> (Vec<f64>, Vec<f64>) {
        if klines.len() < k_period {
            return (vec![f64::NAN; klines.len()], vec![f64::NAN; klines.len()]);
        }

        let mut k_values = vec![f64::NAN; klines.len()];

        for i in (k_period - 1)..klines.len() {
            let slice = &klines[(i + 1 - k_period)..=i];
            let lowest = slice.iter().map(|k| k.low).fold(f64::INFINITY, f64::min);
            let highest = slice
                .iter()
                .map(|k| k.high)
                .fold(f64::NEG_INFINITY, f64::max);

            let range = highest - lowest;
            if range > 0.0 {
                k_values[i] = (klines[i].close - lowest) / range * 100.0;
            } else {
                k_values[i] = 50.0;
            }
        }

        let d_values = Self::sma(&k_values, d_period);

        (k_values, d_values)
    }

    /// Calculate rolling standard deviation.
    pub fn rolling_std(prices: &[f64], period: usize) -> Vec<f64> {
        let sma = Self::sma(prices, period);
        let mut result = vec![f64::NAN; prices.len()];

        for i in (period - 1)..prices.len() {
            let mean = sma[i];
            let slice = &prices[(i + 1 - period)..=i];
            let variance: f64 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Calculate returns.
    pub fn returns(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; prices.len()];

        for i in period..prices.len() {
            if prices[i - period] != 0.0 {
                result[i] = (prices[i] - prices[i - period]) / prices[i - period];
            }
        }

        result
    }

    /// Calculate log returns.
    pub fn log_returns(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; prices.len()];

        for i in period..prices.len() {
            if prices[i - period] > 0.0 && prices[i] > 0.0 {
                result[i] = (prices[i] / prices[i - period]).ln();
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
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::sma(&prices, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let rsi = TechnicalIndicators::rsi(&prices, 14);

        // Consistently rising prices should give high RSI
        assert!(rsi[14] > 70.0);
    }
}
