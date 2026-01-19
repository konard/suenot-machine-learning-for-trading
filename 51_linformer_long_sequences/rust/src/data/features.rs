//! Technical feature calculation for market data.

use ndarray::{Array1, Array2, s};

/// Technical features calculator.
pub struct TechnicalFeatures;

impl TechnicalFeatures {
    /// Calculate Simple Moving Average (SMA).
    pub fn sma(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 >= period {
                let sum: f64 = prices.slice(s![i + 1 - period..=i]).sum();
                result[i] = sum / period as f64;
            } else {
                result[i] = prices.slice(s![0..=i]).mean().unwrap_or(0.0);
            }
        }

        result
    }

    /// Calculate Exponential Moving Average (EMA).
    pub fn ema(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);
        let alpha = 2.0 / (period as f64 + 1.0);

        result[0] = prices[0];
        for i in 1..n {
            result[i] = alpha * prices[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate Relative Strength Index (RSI).
    pub fn rsi(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        if n < 2 {
            return result;
        }

        // Calculate price changes
        let mut gains = Vec::with_capacity(n - 1);
        let mut losses = Vec::with_capacity(n - 1);

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        // Calculate RSI using smoothed averages
        let mut avg_gain = gains.iter().take(period).sum::<f64>() / period as f64;
        let mut avg_loss = losses.iter().take(period).sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i - 1]) / period as f64;
                avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i - 1]) / period as f64;
            }

            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence).
    /// Returns (macd_line, signal_line, histogram)
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

    /// Calculate Bollinger Bands.
    /// Returns (upper_band, middle_band, lower_band)
    pub fn bollinger_bands(
        prices: &Array1<f64>,
        period: usize,
        std_dev: f64,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let middle = Self::sma(prices, period);
        let n = prices.len();

        let mut upper = Array1::zeros(n);
        let mut lower = Array1::zeros(n);

        for i in 0..n {
            let start = if i + 1 >= period { i + 1 - period } else { 0 };
            let window = prices.slice(s![start..=i]);

            let mean = window.mean().unwrap_or(0.0);
            let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64;
            let std = variance.sqrt();

            upper[i] = middle[i] + std_dev * std;
            lower[i] = middle[i] - std_dev * std;
        }

        (upper, middle, lower)
    }

    /// Calculate historical volatility.
    pub fn volatility(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        if n < 2 {
            return result;
        }

        // Calculate log returns
        let mut log_returns = Vec::with_capacity(n - 1);
        for i in 1..n {
            if prices[i - 1] > 0.0 && prices[i] > 0.0 {
                log_returns.push((prices[i] / prices[i - 1]).ln());
            } else {
                log_returns.push(0.0);
            }
        }

        // Calculate rolling standard deviation of returns
        for i in period..n {
            let window = &log_returns[i - period..i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Calculate price momentum.
    pub fn momentum(prices: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = prices.len();
        let mut result = Array1::zeros(n);

        for i in period..n {
            if prices[i - period] != 0.0 {
                result[i] = (prices[i] - prices[i - period]) / prices[i - period];
            }
        }

        result
    }

    /// Calculate all features and return as a matrix.
    /// Returns matrix with columns: [sma_20, ema_20, rsi_14, macd, vol_20, mom_10]
    pub fn calculate_all(prices: &Array1<f64>) -> Array2<f64> {
        let n = prices.len();
        let mut features = Array2::zeros((n, 6));

        let sma_20 = Self::sma(prices, 20);
        let ema_20 = Self::ema(prices, 20);
        let rsi_14 = Self::rsi(prices, 14);
        let (macd_line, _, _) = Self::macd(prices, 12, 26, 9);
        let volatility = Self::volatility(prices, 20);
        let momentum = Self::momentum(prices, 10);

        for i in 0..n {
            features[[i, 0]] = sma_20[i];
            features[[i, 1]] = ema_20[i];
            features[[i, 2]] = rsi_14[i];
            features[[i, 3]] = macd_line[i];
            features[[i, 4]] = volatility[i];
            features[[i, 5]] = momentum[i];
        }

        features
    }

    /// Normalize features using z-score normalization.
    pub fn normalize_zscore(features: &Array2<f64>) -> Array2<f64> {
        let (n_rows, n_cols) = features.dim();
        let mut normalized = Array2::zeros((n_rows, n_cols));

        for j in 0..n_cols {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = {
                let variance: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_rows as f64;
                variance.sqrt()
            };

            for i in 0..n_rows {
                if std > 0.0 {
                    normalized[[i, j]] = (features[[i, j]] - mean) / std;
                } else {
                    normalized[[i, j]] = 0.0;
                }
            }
        }

        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sma() {
        let prices = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalFeatures::sma(&prices, 3);
        assert!((sma[2] - 2.0).abs() < 0.01);
        assert!((sma[4] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_ema() {
        let prices = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = TechnicalFeatures::ema(&prices, 3);
        assert!(ema[4] > ema[0]);
    }

    #[test]
    fn test_rsi_bounds() {
        let prices = array![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0,
                           107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0];
        let rsi = TechnicalFeatures::rsi(&prices, 14);

        for val in rsi.iter() {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_bollinger_bands() {
        let prices = array![100.0, 101.0, 99.0, 100.5, 102.0, 101.0, 100.0];
        let (upper, middle, lower) = TechnicalFeatures::bollinger_bands(&prices, 5, 2.0);

        for i in 0..prices.len() {
            assert!(upper[i] >= middle[i]);
            assert!(middle[i] >= lower[i]);
        }
    }
}
