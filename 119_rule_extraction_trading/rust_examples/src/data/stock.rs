//! Stock market data utilities and feature computation.
//!
//! Provides functions for:
//! - Computing technical indicators (RSI, MACD, SMA, etc.)
//! - Feature engineering for trading models

use serde::{Deserialize, Serialize};

/// Stock market OHLCV data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockData {
    /// Date/time of the bar
    pub date: String,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
}

impl StockData {
    /// Calculate returns from open to close.
    pub fn returns(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open
    }
}

/// Compute Relative Strength Index (RSI).
///
/// RSI measures the speed and magnitude of recent price changes
/// to evaluate overbought or oversold conditions.
///
/// # Arguments
/// * `prices` - Vector of closing prices
/// * `period` - Lookback period (typically 14)
///
/// # Returns
/// Vector of RSI values (0-100). First `period` values will be NaN.
pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![f64::NAN; prices.len()];
    }

    let mut rsi = vec![f64::NAN; prices.len()];

    // Calculate price changes
    let changes: Vec<f64> = prices
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    // Initialize average gains and losses
    let mut avg_gain: f64 = changes[..period]
        .iter()
        .filter(|&&c| c > 0.0)
        .sum::<f64>()
        / period as f64;

    let mut avg_loss: f64 = changes[..period]
        .iter()
        .filter(|&&c| c < 0.0)
        .map(|c| c.abs())
        .sum::<f64>()
        / period as f64;

    // Calculate first RSI
    if avg_loss == 0.0 {
        rsi[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Calculate remaining RSI values using smoothed averages
    for i in period..changes.len() {
        let change = changes[i];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };

        avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;

        if avg_loss == 0.0 {
            rsi[i + 1] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    rsi
}

/// Compute Simple Moving Average (SMA).
///
/// # Arguments
/// * `prices` - Vector of prices
/// * `period` - Lookback period
///
/// # Returns
/// Vector of SMA values. First `period-1` values will be NaN.
pub fn compute_sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![f64::NAN; prices.len()];
    }

    let mut sma = vec![f64::NAN; prices.len()];

    for i in (period - 1)..prices.len() {
        let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
        sma[i] = sum / period as f64;
    }

    sma
}

/// Compute Exponential Moving Average (EMA).
///
/// # Arguments
/// * `prices` - Vector of prices
/// * `period` - Lookback period
///
/// # Returns
/// Vector of EMA values. First `period-1` values will be NaN.
pub fn compute_ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![f64::NAN; prices.len()];
    }

    let mut ema = vec![f64::NAN; prices.len()];
    let multiplier = 2.0 / (period + 1) as f64;

    // Initialize with SMA
    let initial_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
    ema[period - 1] = initial_sma;

    // Calculate EMA
    for i in period..prices.len() {
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    ema
}

/// Compute MACD (Moving Average Convergence Divergence).
///
/// # Arguments
/// * `prices` - Vector of closing prices
/// * `fast_period` - Fast EMA period (typically 12)
/// * `slow_period` - Slow EMA period (typically 26)
/// * `signal_period` - Signal line period (typically 9)
///
/// # Returns
/// Tuple of (MACD line, Signal line, Histogram)
pub fn compute_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = compute_ema(prices, fast_period);
    let slow_ema = compute_ema(prices, slow_period);

    // MACD line = Fast EMA - Slow EMA
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

    // Signal line = EMA of MACD line
    let valid_macd: Vec<f64> = macd_line.iter().filter(|x| !x.is_nan()).copied().collect();
    let signal_ema = compute_ema(&valid_macd, signal_period);

    // Align signal line with original length
    let mut signal_line = vec![f64::NAN; prices.len()];
    let start_idx = macd_line.iter().position(|x| !x.is_nan()).unwrap_or(0);
    for (i, &val) in signal_ema.iter().enumerate() {
        if start_idx + i < signal_line.len() {
            signal_line[start_idx + i] = val;
        }
    }

    // Histogram = MACD line - Signal line
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

/// Compute rolling volatility (standard deviation of returns).
///
/// # Arguments
/// * `returns` - Vector of returns
/// * `period` - Lookback period
///
/// # Returns
/// Vector of volatility values.
pub fn compute_volatility(returns: &[f64], period: usize) -> Vec<f64> {
    if returns.len() < period {
        return vec![f64::NAN; returns.len()];
    }

    let mut volatility = vec![f64::NAN; returns.len()];

    for i in (period - 1)..returns.len() {
        let window = &returns[(i + 1 - period)..=i];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (period - 1) as f64;
        volatility[i] = variance.sqrt();
    }

    volatility
}

/// Compute all features from price data.
///
/// # Arguments
/// * `prices` - Vector of closing prices
///
/// # Returns
/// Vector of feature vectors for each time step.
/// Features: [RSI, MACD, SMA_ratio, Volatility, Volume_change]
pub fn compute_features(data: &[StockData]) -> (Vec<Vec<f64>>, Vec<String>) {
    let prices: Vec<f64> = data.iter().map(|d| d.close).collect();
    let volumes: Vec<f64> = data.iter().map(|d| d.volume).collect();

    // Compute indicators
    let rsi = compute_rsi(&prices, 14);
    let (macd, _, _) = compute_macd(&prices, 12, 26, 9);
    let sma_20 = compute_sma(&prices, 20);
    let sma_50 = compute_sma(&prices, 50);

    // Compute returns and volatility
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();
    let mut padded_returns = vec![0.0];
    padded_returns.extend(returns);

    let volatility = compute_volatility(&padded_returns, 20);

    // Compute volume change
    let volume_change: Vec<f64> = volumes
        .windows(2)
        .map(|w| {
            if w[0] == 0.0 {
                0.0
            } else {
                (w[1] - w[0]) / w[0]
            }
        })
        .collect();
    let mut padded_volume_change = vec![0.0];
    padded_volume_change.extend(volume_change);

    // Build feature vectors
    let mut features = Vec::new();
    for i in 0..prices.len() {
        let sma_ratio = if sma_50[i].is_nan() || sma_50[i] == 0.0 {
            1.0
        } else if sma_20[i].is_nan() {
            1.0
        } else {
            sma_20[i] / sma_50[i]
        };

        features.push(vec![
            if rsi[i].is_nan() { 50.0 } else { rsi[i] },
            if macd[i].is_nan() { 0.0 } else { macd[i] },
            sma_ratio,
            if volatility[i].is_nan() {
                0.01
            } else {
                volatility[i]
            },
            padded_volume_change[i],
        ]);
    }

    let feature_names = vec![
        "RSI".to_string(),
        "MACD".to_string(),
        "SMA_Ratio".to_string(),
        "Volatility".to_string(),
        "Volume_Change".to_string(),
    ];

    (features, feature_names)
}

/// Generate target labels for trading (next period direction).
///
/// # Arguments
/// * `prices` - Vector of closing prices
///
/// # Returns
/// Vector of labels: 1 for up, -1 for down, 0 for unchanged
pub fn generate_labels(prices: &[f64]) -> Vec<i32> {
    let mut labels = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        if i + 1 >= prices.len() {
            labels.push(0);
        } else {
            let change = prices[i + 1] - prices[i];
            if change > 0.0 {
                labels.push(1);
            } else if change < 0.0 {
                labels.push(-1);
            } else {
                labels.push(0);
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = compute_sma(&prices, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 0.0001); // (1+2+3)/3 = 2
        assert!((sma[3] - 3.0).abs() < 0.0001); // (2+3+4)/3 = 3
        assert!((sma[4] - 4.0).abs() < 0.0001); // (3+4+5)/3 = 4
    }

    #[test]
    fn test_compute_rsi() {
        // RSI should be 100 when all gains, 0 when all losses
        let up_prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let rsi = compute_rsi(&up_prices, 14);
        assert!(rsi[14] > 95.0); // Should be close to 100

        let down_prices: Vec<f64> = (0..16).rev().map(|x| x as f64 + 1.0).collect();
        let rsi_down = compute_rsi(&down_prices, 14);
        assert!(rsi_down[14] < 5.0); // Should be close to 0
    }

    #[test]
    fn test_generate_labels() {
        let prices = vec![100.0, 105.0, 103.0, 103.0, 110.0];
        let labels = generate_labels(&prices);

        assert_eq!(labels[0], 1);  // 100 -> 105 (up)
        assert_eq!(labels[1], -1); // 105 -> 103 (down)
        assert_eq!(labels[2], 0);  // 103 -> 103 (unchanged)
        assert_eq!(labels[3], 1);  // 103 -> 110 (up)
    }
}
