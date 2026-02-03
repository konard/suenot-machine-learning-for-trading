//! Technical indicators and feature engineering for financial time series.

use crate::api::bybit::Candle;
use ndarray::{Array1, Array2};

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Simple Moving Average
    pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            sum += data[i] - data[i - period];
            result.push(sum / period as f64);
        }

        result
    }

    /// Exponential Moving Average
    pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period {
            return vec![f64::NAN; data.len()];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![f64::NAN; period - 1];

        // Initial EMA is SMA
        let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result.push(initial_sma);

        let mut prev_ema = initial_sma;
        for &value in &data[period..] {
            let ema = (value - prev_ema) * multiplier + prev_ema;
            result.push(ema);
            prev_ema = ema;
        }

        result
    }

    /// Relative Strength Index
    pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period + 1 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period];

        // Calculate price changes
        let changes: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();

        // Calculate initial average gain and loss
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for &change in &changes[..period] {
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += change.abs();
            }
        }

        avg_gain /= period as f64;
        avg_loss /= period as f64;

        // Calculate RSI
        let rs = if avg_loss == 0.0 {
            100.0
        } else {
            avg_gain / avg_loss
        };
        result.push(100.0 - (100.0 / (1.0 + rs)));

        // Smoothed RSI for remaining periods
        for &change in &changes[period..] {
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, change.abs())
            };

            avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;

            let rs = if avg_loss == 0.0 {
                100.0
            } else {
                avg_gain / avg_loss
            };
            result.push(100.0 - (100.0 / (1.0 + rs)));
        }

        result
    }

    /// MACD (Moving Average Convergence Divergence)
    /// Returns (macd_line, signal_line, histogram)
    pub fn macd(
        data: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let fast_ema = Self::ema(data, fast_period);
        let slow_ema = Self::ema(data, slow_period);

        // MACD line
        let macd_line: Vec<f64> = fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(&f, &s)| f - s)
            .collect();

        // Signal line (EMA of MACD)
        let signal_line = Self::ema(&macd_line, signal_period);

        // Histogram
        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(&m, &s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Bollinger Bands
    /// Returns (upper_band, middle_band, lower_band)
    pub fn bollinger_bands(
        data: &[f64],
        period: usize,
        std_dev: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = Self::sma(data, period);

        let mut upper = vec![f64::NAN; data.len()];
        let mut lower = vec![f64::NAN; data.len()];

        for i in period - 1..data.len() {
            let window = &data[i + 1 - period..=i];
            let mean = middle[i];

            if mean.is_nan() {
                continue;
            }

            let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }

        (upper, middle, lower)
    }

    /// Average True Range
    pub fn atr(candles: &[Candle], period: usize) -> Vec<f64> {
        if candles.len() < 2 {
            return vec![f64::NAN; candles.len()];
        }

        // Calculate true range
        let mut tr = vec![f64::NAN];
        for i in 1..candles.len() {
            tr.push(candles[i].true_range(candles[i - 1].close));
        }

        // Calculate ATR using EMA
        Self::ema(&tr, period)
    }

    /// Historical Volatility
    pub fn volatility(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < 2 {
            return vec![f64::NAN; data.len()];
        }

        // Calculate log returns
        let returns: Vec<f64> = data
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let mut result = vec![f64::NAN; period];

        for i in period..returns.len() + 1 {
            let window = &returns[i - period..i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / period as f64;
            // Annualize (assuming 252 trading days)
            result.push(variance.sqrt() * (252.0_f64).sqrt());
        }

        result
    }

    /// Returns (simple price change)
    pub fn returns(data: &[f64]) -> Vec<f64> {
        let mut result = vec![f64::NAN];
        for i in 1..data.len() {
            result.push((data[i] - data[i - 1]) / data[i - 1]);
        }
        result
    }

    /// Log returns
    pub fn log_returns(data: &[f64]) -> Vec<f64> {
        let mut result = vec![f64::NAN];
        for i in 1..data.len() {
            result.push((data[i] / data[i - 1]).ln());
        }
        result
    }
}

/// Prepare features from candlestick data
pub fn prepare_features(candles: &[Candle]) -> Array2<f64> {
    if candles.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n = candles.len();

    // Extract OHLCV
    let open: Vec<f64> = candles.iter().map(|c| c.open).collect();
    let high: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let low: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let close: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volume: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // Calculate indicators
    let sma_10 = TechnicalIndicators::sma(&close, 10);
    let sma_20 = TechnicalIndicators::sma(&close, 20);
    let ema_10 = TechnicalIndicators::ema(&close, 10);
    let ema_20 = TechnicalIndicators::ema(&close, 20);
    let rsi = TechnicalIndicators::rsi(&close, 14);
    let (macd, macd_signal, macd_hist) = TechnicalIndicators::macd(&close, 12, 26, 9);
    let (bb_upper, bb_middle, bb_lower) = TechnicalIndicators::bollinger_bands(&close, 20, 2.0);
    let atr = TechnicalIndicators::atr(candles, 14);
    let volatility = TechnicalIndicators::volatility(&close, 20);
    let returns = TechnicalIndicators::returns(&close);
    let log_returns = TechnicalIndicators::log_returns(&close);

    // Bollinger Band width
    let bb_width: Vec<f64> = bb_upper
        .iter()
        .zip(bb_lower.iter())
        .zip(bb_middle.iter())
        .map(|((&u, &l), &m)| {
            if m.is_nan() || m == 0.0 {
                f64::NAN
            } else {
                (u - l) / m
            }
        })
        .collect();

    // Price position within Bollinger Bands
    let price_position: Vec<f64> = close
        .iter()
        .zip(bb_upper.iter())
        .zip(bb_lower.iter())
        .map(|((&c, &u), &l)| {
            if u.is_nan() || l.is_nan() || (u - l) == 0.0 {
                f64::NAN
            } else {
                (c - l) / (u - l)
            }
        })
        .collect();

    // Number of features
    let n_features = 17;

    // Create feature matrix
    let mut features = Array2::zeros((n, n_features));

    for i in 0..n {
        features[[i, 0]] = open[i];
        features[[i, 1]] = high[i];
        features[[i, 2]] = low[i];
        features[[i, 3]] = close[i];
        features[[i, 4]] = volume[i];
        features[[i, 5]] = sma_10[i];
        features[[i, 6]] = sma_20[i];
        features[[i, 7]] = ema_10[i];
        features[[i, 8]] = ema_20[i];
        features[[i, 9]] = rsi[i];
        features[[i, 10]] = macd[i];
        features[[i, 11]] = macd_signal[i];
        features[[i, 12]] = bb_width[i];
        features[[i, 13]] = atr[i];
        features[[i, 14]] = volatility[i];
        features[[i, 15]] = returns[i];
        features[[i, 16]] = price_position[i];
    }

    features
}

/// Normalize features using z-score normalization
pub fn normalize_features(features: &Array2<f64>) -> (Array2<f64>, Vec<(f64, f64)>) {
    let (n_samples, n_features) = features.dim();
    let mut normalized = Array2::zeros((n_samples, n_features));
    let mut stats = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let column: Vec<f64> = features
            .column(j)
            .iter()
            .filter(|&&x| !x.is_nan())
            .copied()
            .collect();

        if column.is_empty() {
            stats.push((0.0, 1.0));
            continue;
        }

        let mean = column.iter().sum::<f64>() / column.len() as f64;
        let variance: f64 =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
        let std = variance.sqrt().max(1e-8);

        stats.push((mean, std));

        for i in 0..n_samples {
            let value = features[[i, j]];
            normalized[[i, j]] = if value.is_nan() {
                0.0
            } else {
                (value - mean) / std
            };
        }
    }

    (normalized, stats)
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
        assert!((sma[2] - 2.0).abs() < 0.001);
        assert!((sma[3] - 3.0).abs() < 0.001);
        assert!((sma[4] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = TechnicalIndicators::ema(&data, 3);
        assert!(ema[0].is_nan());
        assert!(ema[1].is_nan());
        assert!((ema[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.75, 44.0, 44.5, 44.75, 45.0, 45.25, 45.5, 45.75];
        let rsi = TechnicalIndicators::rsi(&data, 14);
        assert!(rsi.last().is_some());
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 110.0, 99.0, 110.0];
        let returns = TechnicalIndicators::returns(&data);
        assert!(returns[0].is_nan());
        assert!((returns[1] - 0.1).abs() < 0.001);
        assert!((returns[2] - (-0.1)).abs() < 0.001);
    }
}
