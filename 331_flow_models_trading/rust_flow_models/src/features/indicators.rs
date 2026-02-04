//! Technical indicators for feature engineering

use ndarray::{Array1, Array2};

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate simple returns
    pub fn returns(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        for i in period..n {
            result[i] = (prices[i] - prices[i - period]) / prices[i - period];
        }

        result
    }

    /// Calculate log returns
    pub fn log_returns(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        for i in period..n {
            result[i] = (prices[i] / prices[i - period]).ln();
        }

        result
    }

    /// Calculate simple moving average
    pub fn sma(data: &Array1<f64>, window: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in window - 1..n {
            let sum: f64 = data.slice(ndarray::s![i + 1 - window..=i]).sum();
            result[i] = sum / window as f64;
        }

        result
    }

    /// Calculate exponential moving average
    pub fn ema(data: &Array1<f64>, span: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        if n == 0 {
            return result;
        }

        let alpha = 2.0 / (span as f64 + 1.0);
        result[0] = data[0];

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(data: &Array1<f64>, window: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in window - 1..n {
            let slice = data.slice(ndarray::s![i + 1 - window..=i]);
            let mean = slice.mean().unwrap_or(0.0);
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        if n < period + 1 {
            return result;
        }

        // Calculate price changes
        let mut gains = Array1::zeros(n);
        let mut losses = Array1::zeros(n);

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate average gains and losses using EMA
        let avg_gains = Self::ema(&gains, period);
        let avg_losses = Self::ema(&losses, period);

        // Calculate RSI
        for i in period..n {
            let rs = if avg_losses[i] > 0.0 {
                avg_gains[i] / avg_losses[i]
            } else {
                100.0
            };
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(
        prices: &Array1<f64>,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let fast_ema = Self::ema(prices, fast_period);
        let slow_ema = Self::ema(prices, slow_period);

        let macd_line = &fast_ema - &slow_ema;
        let signal_line = Self::ema(&macd_line, signal_period);
        let histogram = &macd_line - &signal_line;

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(
        prices: &Array1<f64>,
        window: usize,
        num_std: f64,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let middle = Self::sma(prices, window);
        let std = Self::rolling_std(prices, window);

        let upper = &middle + &(&std * num_std);
        let lower = &middle - &(&std * num_std);

        (upper, middle, lower)
    }

    /// Calculate Bollinger Band position (0 to 1)
    pub fn bollinger_position(prices: &Array1<f64>, window: usize, num_std: f64) -> Array1<f64> {
        let (upper, _, lower) = Self::bollinger_bands(prices, window, num_std);
        let n = prices.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let range = upper[i] - lower[i];
            if range > 0.0 {
                result[i] = (prices[i] - lower[i]) / range;
            } else {
                result[i] = 0.5;
            }
        }

        result
    }

    /// Calculate ATR (Average True Range)
    pub fn atr(high: &Array1<f64>, low: &Array1<f64>, close: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = high.len();
        let mut tr = Array1::zeros(n);

        tr[0] = high[0] - low[0];

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        Self::ema(&tr, period)
    }

    /// Calculate volume ratio (current / average)
    pub fn volume_ratio(volume: &Array1<f64>, window: usize) -> Array1<f64> {
        let avg_volume = Self::sma(volume, window);
        let n = volume.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if avg_volume[i] > 0.0 {
                result[i] = volume[i] / avg_volume[i];
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
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let sma = TechnicalIndicators::sma(&data, 3);

        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let prices = Array1::from_vec(vec![100.0, 110.0, 105.0, 115.0]);
        let returns = TechnicalIndicators::returns(&prices, 1);

        assert!((returns[1] - 0.1).abs() < 1e-10);
        assert!((returns[2] - (-0.0454545)).abs() < 1e-5);
    }

    #[test]
    fn test_rsi() {
        let prices = Array1::from_vec(vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.5, 44.75, 44.5, 44.75,
            45.0, 45.25, 45.5, 45.25, 45.5,
        ]);
        let rsi = TechnicalIndicators::rsi(&prices, 14);

        // RSI should be between 0 and 100
        for i in 14..prices.len() {
            assert!(rsi[i] >= 0.0 && rsi[i] <= 100.0);
        }
    }
}
