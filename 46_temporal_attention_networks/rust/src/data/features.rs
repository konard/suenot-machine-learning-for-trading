//! Feature engineering for TABL model

use crate::api::Kline;
use ndarray::{Array1, Array2, Axis};

/// Container for computed features
#[derive(Debug, Clone)]
pub struct Features {
    /// Feature matrix [time_steps, n_features]
    pub data: Array2<f64>,
    /// Feature names
    pub names: Vec<String>,
    /// Timestamps corresponding to each row
    pub timestamps: Vec<i64>,
}

impl Features {
    /// Get the number of time steps
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if features are empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Normalize features to zero mean and unit variance
    pub fn normalize(&mut self) {
        for col_idx in 0..self.data.ncols() {
            let col = self.data.column(col_idx);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);

            if std > 1e-8 {
                for row_idx in 0..self.data.nrows() {
                    self.data[[row_idx, col_idx]] = (self.data[[row_idx, col_idx]] - mean) / std;
                }
            }
        }
    }

    /// Get a slice of features for a time window
    pub fn window(&self, start: usize, length: usize) -> Option<Array2<f64>> {
        if start + length > self.len() {
            return None;
        }
        Some(self.data.slice(ndarray::s![start..start + length, ..]).to_owned())
    }
}

/// Compute log returns
fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute RSI (Relative Strength Index)
fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![50.0; prices.len()];
    }

    let mut rsi = vec![50.0; period];
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // Initial averages
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    // Calculate first RSI
    let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
    rsi.push(100.0 - 100.0 / (1.0 + rs));

    // Subsequent values using exponential smoothing
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };

        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
        rsi.push(100.0 - 100.0 / (1.0 + rs));
    }

    rsi
}

/// Compute MACD (Moving Average Convergence Divergence)
fn compute_macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>) {
    let ema_fast = compute_ema(prices, fast);
    let ema_slow = compute_ema(prices, slow);

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = compute_ema(&macd_line, signal);

    (macd_line, signal_line)
}

/// Compute Exponential Moving Average
fn compute_ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![];
    }

    let mut ema = vec![0.0; data.len()];
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Start with SMA
    let sma: f64 = data.iter().take(period).sum::<f64>() / period as f64;
    ema[period - 1] = sma;

    // Calculate EMA
    for i in period..data.len() {
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    // Fill initial values
    for i in 0..period - 1 {
        ema[i] = data[..=i].iter().sum::<f64>() / (i + 1) as f64;
    }

    ema
}

/// Compute Bollinger Bands
fn compute_bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut upper = vec![0.0; prices.len()];
    let mut middle = vec![0.0; prices.len()];
    let mut lower = vec![0.0; prices.len()];

    for i in 0..prices.len() {
        let start = if i >= period { i - period + 1 } else { 0 };
        let window = &prices[start..=i];

        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
        let std = variance.sqrt();

        middle[i] = mean;
        upper[i] = mean + num_std * std;
        lower[i] = mean - num_std * std;
    }

    (upper, middle, lower)
}

/// Compute ATR (Average True Range)
fn compute_atr(klines: &[Kline], period: usize) -> Vec<f64> {
    if klines.is_empty() {
        return vec![];
    }

    let mut tr = vec![0.0; klines.len()];

    // First TR is just high - low
    tr[0] = klines[0].high - klines[0].low;

    // Subsequent TRs
    for i in 1..klines.len() {
        let high_low = klines[i].high - klines[i].low;
        let high_close = (klines[i].high - klines[i - 1].close).abs();
        let low_close = (klines[i].low - klines[i - 1].close).abs();
        tr[i] = high_low.max(high_close).max(low_close);
    }

    // Compute ATR as EMA of TR
    compute_ema(&tr, period)
}

/// Compute volume-related features
fn compute_volume_features(klines: &[Kline], period: usize) -> (Vec<f64>, Vec<f64>) {
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    // Volume SMA
    let volume_sma = compute_sma(&volumes, period);

    // Volume ratio (current / average)
    let volume_ratio: Vec<f64> = volumes
        .iter()
        .zip(volume_sma.iter())
        .map(|(v, sma)| if *sma > 0.0 { v / sma } else { 1.0 })
        .collect();

    (volume_sma, volume_ratio)
}

/// Compute Simple Moving Average
fn compute_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut sma = vec![0.0; data.len()];

    for i in 0..data.len() {
        let start = if i >= period { i - period + 1 } else { 0 };
        let window = &data[start..=i];
        sma[i] = window.iter().sum::<f64>() / window.len() as f64;
    }

    sma
}

/// Prepare features from kline data
///
/// # Arguments
/// * `klines` - Vector of OHLCV candlestick data
///
/// # Returns
/// * `Features` struct containing computed features
pub fn prepare_features(klines: &[Kline]) -> Features {
    if klines.is_empty() {
        return Features {
            data: Array2::zeros((0, 0)),
            names: vec![],
            timestamps: vec![],
        };
    }

    let close_prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let timestamps: Vec<i64> = klines.iter().map(|k| k.start_time).collect();

    // Compute features
    let log_returns = compute_log_returns(&close_prices);
    let rsi = compute_rsi(&close_prices, 14);
    let (macd_line, signal_line) = compute_macd(&close_prices, 12, 26, 9);
    let (bb_upper, bb_middle, bb_lower) = compute_bollinger_bands(&close_prices, 20, 2.0);
    let atr = compute_atr(klines, 14);
    let (_, volume_ratio) = compute_volume_features(klines, 20);

    // Pad log_returns to match length (first value is 0)
    let mut padded_returns = vec![0.0];
    padded_returns.extend(log_returns);

    // Bollinger Band position (where price is relative to bands)
    let bb_position: Vec<f64> = close_prices
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let range = bb_upper[i] - bb_lower[i];
            if range > 0.0 {
                (p - bb_lower[i]) / range
            } else {
                0.5
            }
        })
        .collect();

    // MACD histogram
    let macd_histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    // Build feature matrix
    let n_samples = klines.len();
    let n_features = 6;
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        data[[i, 0]] = padded_returns[i];
        data[[i, 1]] = rsi[i] / 100.0; // Normalize RSI to [0, 1]
        data[[i, 2]] = macd_histogram[i];
        data[[i, 3]] = bb_position[i];
        data[[i, 4]] = atr[i];
        data[[i, 5]] = volume_ratio[i];
    }

    let names = vec![
        "log_return".to_string(),
        "rsi".to_string(),
        "macd_histogram".to_string(),
        "bb_position".to_string(),
        "atr".to_string(),
        "volume_ratio".to_string(),
    ];

    Features {
        data,
        names,
        timestamps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                start_time: i as i64 * 3600000,
                open: 100.0 + (i as f64 * 0.1).sin(),
                high: 101.0 + (i as f64 * 0.1).sin(),
                low: 99.0 + (i as f64 * 0.1).sin(),
                close: 100.5 + (i as f64 * 0.1).sin(),
                volume: 1000.0 + (i as f64 * 10.0),
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_prepare_features() {
        let klines = create_test_klines(100);
        let features = prepare_features(&klines);

        assert_eq!(features.len(), 100);
        assert_eq!(features.n_features(), 6);
        assert_eq!(features.names.len(), 6);
    }

    #[test]
    fn test_compute_rsi() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let rsi = compute_rsi(&prices, 14);

        assert_eq!(rsi.len(), prices.len());
        assert!(rsi.iter().all(|&r| r >= 0.0 && r <= 100.0));
    }

    #[test]
    fn test_compute_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ema = compute_ema(&data, 3);

        assert_eq!(ema.len(), data.len());
        // EMA should be close to recent values
        assert!(ema[9] > ema[0]);
    }

    #[test]
    fn test_features_window() {
        let klines = create_test_klines(100);
        let features = prepare_features(&klines);

        let window = features.window(10, 20);
        assert!(window.is_some());
        let window = window.unwrap();
        assert_eq!(window.nrows(), 20);
        assert_eq!(window.ncols(), 6);
    }

    #[test]
    fn test_features_normalize() {
        let klines = create_test_klines(100);
        let mut features = prepare_features(&klines);
        features.normalize();

        // Check that normalized columns have approximately zero mean
        // Some columns may have very low variance (like constant ATR from synthetic data)
        // and won't be normalized to avoid division by near-zero
        let mut normalized_count = 0;
        for col_idx in 0..features.n_features() {
            let col = features.data.column(col_idx);
            let mean = col.mean().unwrap();
            // Check if column was normalized (mean near zero) or skipped (has variance < 1e-8)
            if mean.abs() < 0.5 {
                normalized_count += 1;
            }
        }
        // At least some columns should be normalized
        assert!(normalized_count >= 3, "Expected at least 3 normalized columns, got {}", normalized_count);
    }
}
