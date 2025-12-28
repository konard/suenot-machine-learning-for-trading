//! Feature engineering for cryptocurrency price prediction
//!
//! This module provides various technical indicators and features
//! commonly used in cryptocurrency trading strategies.

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering utilities
#[derive(Debug, Default)]
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Create a new feature engineering instance
    pub fn new() -> Self {
        Self
    }

    /// Calculate Simple Moving Average (SMA)
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..prices.len() {
            let sum: f64 = prices[i + 1 - period..=i].iter().sum();
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate Exponential Moving Average (EMA)
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![f64::NAN; period - 1];

        // First EMA is SMA
        let first_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
        result.push(first_sma);

        let mut prev_ema = first_sma;
        for &price in &prices[period..] {
            let ema = (price - prev_ema) * multiplier + prev_ema;
            result.push(ema);
            prev_ema = ema;
        }

        result
    }

    /// Calculate Relative Strength Index (RSI)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![f64::NAN; prices.len()];
        }

        // Calculate price changes
        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        let gains: Vec<f64> = changes.iter().map(|&c| if c > 0.0 { c } else { 0.0 }).collect();
        let losses: Vec<f64> = changes
            .iter()
            .map(|&c| if c < 0.0 { -c } else { 0.0 })
            .collect();

        let mut result = vec![f64::NAN; period];

        // First average
        let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        for i in period..changes.len() {
            if avg_loss > 1e-10 {
                let rs = avg_gain / avg_loss;
                result.push(100.0 - (100.0 / (1.0 + rs)));
            } else {
                result.push(100.0);
            }

            // Smoothed averages
            avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        }

        // Add last value
        if avg_loss > 1e-10 {
            let rs = avg_gain / avg_loss;
            result.push(100.0 - (100.0 / (1.0 + rs)));
        } else {
            result.push(100.0);
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        // MACD line
        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(&f, &s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        // Signal line (EMA of MACD)
        let valid_macd: Vec<f64> = macd_line.iter().filter(|x| !x.is_nan()).cloned().collect();
        let signal_line_valid = Self::ema(&valid_macd, signal);

        let mut signal_line = vec![f64::NAN; macd_line.len() - valid_macd.len()];
        signal_line.extend(signal_line_valid);

        // Histogram
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

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, period);

        let mut upper = vec![f64::NAN; period - 1];
        let mut lower = vec![f64::NAN; period - 1];

        for i in (period - 1)..prices.len() {
            let window = &prices[i + 1 - period..=i];
            let mean = sma[i];
            let variance: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper.push(mean + num_std * std);
            lower.push(mean - num_std * std);
        }

        (upper, sma, lower)
    }

    /// Calculate Average True Range (ATR)
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < 2 {
            return vec![f64::NAN; klines.len()];
        }

        // Calculate True Range
        let mut tr = vec![klines[0].high - klines[0].low];

        for i in 1..klines.len() {
            let high_low = klines[i].high - klines[i].low;
            let high_prev_close = (klines[i].high - klines[i - 1].close).abs();
            let low_prev_close = (klines[i].low - klines[i - 1].close).abs();

            tr.push(high_low.max(high_prev_close).max(low_prev_close));
        }

        // Calculate ATR as EMA of TR
        Self::ema(&tr, period)
    }

    /// Calculate momentum
    pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..prices.len() {
            result.push(prices[i] - prices[i - period]);
        }

        result
    }

    /// Calculate Rate of Change (ROC)
    pub fn roc(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..prices.len() {
            if prices[i - period].abs() > 1e-10 {
                result.push((prices[i] - prices[i - period]) / prices[i - period] * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }

    /// Calculate volatility (rolling standard deviation of returns)
    pub fn volatility(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![f64::NAN; prices.len()];
        }

        // Calculate returns
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

        let mut result = vec![f64::NAN; period];

        for i in period..returns.len() {
            let window = &returns[i - period + 1..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result.push(variance.sqrt());
        }

        // Pad for first price
        result.insert(0, f64::NAN);

        result
    }

    /// Calculate On-Balance Volume (OBV)
    pub fn obv(klines: &[Kline]) -> Vec<f64> {
        if klines.is_empty() {
            return vec![];
        }

        let mut result = vec![0.0];

        for i in 1..klines.len() {
            let prev_obv = result[i - 1];
            let obv = if klines[i].close > klines[i - 1].close {
                prev_obv + klines[i].volume
            } else if klines[i].close < klines[i - 1].close {
                prev_obv - klines[i].volume
            } else {
                prev_obv
            };
            result.push(obv);
        }

        result
    }

    /// Generate all features from kline data
    pub fn generate_features(klines: &[Kline]) -> (Array2<f64>, Vec<String>) {
        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let n = klines.len();

        // Calculate all features
        let returns: Vec<f64> = std::iter::once(f64::NAN)
            .chain(prices.windows(2).map(|w| (w[1] / w[0]) - 1.0))
            .collect();

        let log_returns: Vec<f64> = std::iter::once(f64::NAN)
            .chain(prices.windows(2).map(|w| (w[1] / w[0]).ln()))
            .collect();

        let sma_5 = Self::sma(&prices, 5);
        let sma_10 = Self::sma(&prices, 10);
        let sma_20 = Self::sma(&prices, 20);

        let ema_5 = Self::ema(&prices, 5);
        let ema_10 = Self::ema(&prices, 10);
        let ema_20 = Self::ema(&prices, 20);

        let rsi_14 = Self::rsi(&prices, 14);

        let (macd_line, macd_signal, macd_hist) = Self::macd(&prices, 12, 26, 9);

        let (bb_upper, bb_middle, bb_lower) = Self::bollinger_bands(&prices, 20, 2.0);

        let atr_14 = Self::atr(klines, 14);

        let momentum_10 = Self::momentum(&prices, 10);
        let roc_10 = Self::roc(&prices, 10);
        let volatility_20 = Self::volatility(&prices, 20);

        let obv = Self::obv(klines);

        // Volume features
        let volume: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let volume_sma_10 = Self::sma(&volume, 10);
        let volume_ratio: Vec<f64> = volume
            .iter()
            .zip(volume_sma_10.iter())
            .map(|(&v, &sma)| {
                if sma > 1e-10 && !sma.is_nan() {
                    v / sma
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Price relative to SMAs
        let price_to_sma_5: Vec<f64> = prices
            .iter()
            .zip(sma_5.iter())
            .map(|(&p, &s)| if !s.is_nan() { p / s - 1.0 } else { f64::NAN })
            .collect();

        let price_to_sma_20: Vec<f64> = prices
            .iter()
            .zip(sma_20.iter())
            .map(|(&p, &s)| if !s.is_nan() { p / s - 1.0 } else { f64::NAN })
            .collect();

        // BB position
        let bb_position: Vec<f64> = prices
            .iter()
            .zip(bb_upper.iter().zip(bb_lower.iter()))
            .map(|(&p, (&u, &l))| {
                if !u.is_nan() && !l.is_nan() && (u - l).abs() > 1e-10 {
                    (p - l) / (u - l)
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Lagged returns
        let ret_lag_1: Vec<f64> = std::iter::once(f64::NAN)
            .chain(returns[..n - 1].iter().cloned())
            .collect();

        let ret_lag_2: Vec<f64> = std::iter::repeat(f64::NAN)
            .take(2)
            .chain(returns[..n - 2].iter().cloned())
            .collect();

        let ret_lag_3: Vec<f64> = std::iter::repeat(f64::NAN)
            .take(3)
            .chain(returns[..n - 3].iter().cloned())
            .collect();

        // High-low range ratio
        let hl_range: Vec<f64> = klines
            .iter()
            .map(|k| {
                if k.open > 1e-10 {
                    (k.high - k.low) / k.open
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Build feature matrix
        let feature_names = vec![
            "returns".to_string(),
            "log_returns".to_string(),
            "sma_5".to_string(),
            "sma_10".to_string(),
            "sma_20".to_string(),
            "ema_5".to_string(),
            "ema_10".to_string(),
            "ema_20".to_string(),
            "rsi_14".to_string(),
            "macd_line".to_string(),
            "macd_signal".to_string(),
            "macd_hist".to_string(),
            "bb_upper".to_string(),
            "bb_middle".to_string(),
            "bb_lower".to_string(),
            "atr_14".to_string(),
            "momentum_10".to_string(),
            "roc_10".to_string(),
            "volatility_20".to_string(),
            "obv".to_string(),
            "volume_ratio".to_string(),
            "price_to_sma_5".to_string(),
            "price_to_sma_20".to_string(),
            "bb_position".to_string(),
            "ret_lag_1".to_string(),
            "ret_lag_2".to_string(),
            "ret_lag_3".to_string(),
            "hl_range".to_string(),
        ];

        let all_features: Vec<Vec<f64>> = vec![
            returns,
            log_returns,
            sma_5,
            sma_10,
            sma_20,
            ema_5,
            ema_10,
            ema_20,
            rsi_14,
            macd_line,
            macd_signal,
            macd_hist,
            bb_upper,
            bb_middle,
            bb_lower,
            atr_14,
            momentum_10,
            roc_10,
            volatility_20,
            obv,
            volume_ratio,
            price_to_sma_5,
            price_to_sma_20,
            bb_position,
            ret_lag_1,
            ret_lag_2,
            ret_lag_3,
            hl_range,
        ];

        let n_features = all_features.len();

        let flat: Vec<f64> = (0..n)
            .flat_map(|i| all_features.iter().map(move |f| f[i]))
            .collect();

        let features = Array2::from_shape_vec((n, n_features), flat).unwrap();

        (features, feature_names)
    }

    /// Create target variable (future returns)
    pub fn create_target(klines: &[Kline], forward_periods: usize) -> Array1<f64> {
        let n = klines.len();
        let mut target = vec![f64::NAN; n];

        for i in 0..(n - forward_periods) {
            target[i] = (klines[i + forward_periods].close / klines[i].close) - 1.0;
        }

        Array1::from_vec(target)
    }

    /// Create binary target (1 if price goes up, 0 otherwise)
    pub fn create_binary_target(klines: &[Kline], forward_periods: usize) -> Array1<f64> {
        let n = klines.len();
        let mut target = vec![f64::NAN; n];

        for i in 0..(n - forward_periods) {
            target[i] = if klines[i + forward_periods].close > klines[i].close {
                1.0
            } else {
                0.0
            };
        }

        Array1::from_vec(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = FeatureEngineering::sma(&prices, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.25, 44.75, 44.5, 44.0, 43.5, 43.25, 43.5];
        let rsi = FeatureEngineering::rsi(&prices, 14);

        // RSI should be between 0 and 100
        for &r in &rsi {
            if !r.is_nan() {
                assert!(r >= 0.0 && r <= 100.0);
            }
        }
    }

    #[test]
    fn test_momentum() {
        let prices = vec![100.0, 102.0, 105.0, 103.0, 108.0];
        let mom = FeatureEngineering::momentum(&prices, 2);

        assert!(mom[0].is_nan());
        assert!(mom[1].is_nan());
        assert!((mom[2] - 5.0).abs() < 1e-10);  // 105 - 100
        assert!((mom[3] - 1.0).abs() < 1e-10);  // 103 - 102
        assert!((mom[4] - 3.0).abs() < 1e-10);  // 108 - 105
    }
}
