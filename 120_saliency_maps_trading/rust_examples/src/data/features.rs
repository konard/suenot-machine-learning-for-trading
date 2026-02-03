//! Feature engineering for trading data

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering utilities
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Compute returns from price series
    pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices
            .windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect()
    }

    /// Compute simple moving average
    pub fn sma(prices: &[f64], window: usize) -> Vec<f64> {
        if prices.len() < window {
            return vec![];
        }

        prices
            .windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Compute exponential moving average
    pub fn ema(prices: &[f64], window: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let alpha = 2.0 / (window as f64 + 1.0);
        let mut ema = Vec::with_capacity(prices.len());
        ema.push(prices[0]);

        for i in 1..prices.len() {
            let value = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
            ema.push(value);
        }

        ema
    }

    /// Compute RSI (Relative Strength Index)
    pub fn rsi(prices: &[f64], window: usize) -> Vec<f64> {
        if prices.len() < window + 1 {
            return vec![];
        }

        let returns = Self::compute_returns(prices);
        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for r in &returns {
            if *r > 0.0 {
                gains.push(*r);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-r);
            }
        }

        let mut rsi_values = Vec::new();

        for i in (window - 1)..returns.len() {
            let start = i + 1 - window;
            let avg_gain: f64 = gains[start..=i].iter().sum::<f64>() / window as f64;
            let avg_loss: f64 = losses[start..=i].iter().sum::<f64>() / window as f64;

            let rsi = if avg_loss < 1e-10 {
                100.0
            } else {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            };

            rsi_values.push(rsi);
        }

        rsi_values
    }

    /// Compute MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        if ema_fast.len() != ema_slow.len() {
            return (vec![], vec![]);
        }

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = Self::ema(&macd_line, signal);

        (macd_line, signal_line)
    }

    /// Compute Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if prices.len() < window {
            return (vec![], vec![], vec![]);
        }

        let mut upper = Vec::new();
        let mut middle = Vec::new();
        let mut lower = Vec::new();

        for i in (window - 1)..prices.len() {
            let start = i + 1 - window;
            let slice = &prices[start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            let std = variance.sqrt();

            middle.push(mean);
            upper.push(mean + num_std * std);
            lower.push(mean - num_std * std);
        }

        (upper, middle, lower)
    }

    /// Compute volatility (rolling standard deviation of returns)
    pub fn volatility(prices: &[f64], window: usize) -> Vec<f64> {
        let returns = Self::compute_returns(prices);

        if returns.len() < window {
            return vec![];
        }

        returns
            .windows(window)
            .map(|w| {
                let mean: f64 = w.iter().sum::<f64>() / window as f64;
                let variance: f64 = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Create full feature matrix from klines
    pub fn create_features(klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Compute all indicators
        let returns = Self::compute_returns(&closes);
        let sma_5 = Self::sma(&closes, 5);
        let sma_20 = Self::sma(&closes, 20);
        let rsi = Self::rsi(&closes, 14);
        let (macd_line, macd_signal) = Self::macd(&closes, 12, 26, 9);
        let volatility = Self::volatility(&closes, 20);

        // Find minimum length
        let min_len = [
            n,
            returns.len(),
            sma_5.len(),
            sma_20.len(),
            rsi.len(),
            macd_line.len(),
            volatility.len(),
        ]
        .into_iter()
        .min()
        .unwrap_or(0);

        if min_len == 0 {
            return Array2::zeros((0, 10));
        }

        // Create feature matrix (aligned to end)
        let num_features = 10;
        let mut features = Array2::zeros((min_len, num_features));

        let offset = n - min_len;

        for i in 0..min_len {
            let idx = offset + i;
            features[[i, 0]] = klines[idx].open;
            features[[i, 1]] = klines[idx].high;
            features[[i, 2]] = klines[idx].low;
            features[[i, 3]] = klines[idx].close;
            features[[i, 4]] = klines[idx].volume;

            // Align computed features
            if i < returns.len() {
                features[[i, 5]] = returns[returns.len() - min_len + i];
            }
            if i < sma_5.len() {
                features[[i, 6]] = sma_5[sma_5.len() - min_len + i];
            }
            if i < rsi.len() {
                features[[i, 7]] = rsi[rsi.len() - min_len + i];
            }
            if i < macd_line.len() {
                features[[i, 8]] = macd_line[macd_line.len() - min_len + i];
            }
            if i < volatility.len() {
                features[[i, 9]] = volatility[volatility.len() - min_len + i];
            }
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = FeatureEngineering::sma(&prices, 3);
        assert_eq!(sma.len(), 3);
        assert!((sma[0] - 2.0).abs() < 1e-10);
        assert!((sma[1] - 3.0).abs() < 1e-10);
        assert!((sma[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let rsi = FeatureEngineering::rsi(&prices, 14);
        assert!(!rsi.is_empty());
        for r in &rsi {
            assert!(*r >= 0.0 && *r <= 100.0);
        }
    }
}
