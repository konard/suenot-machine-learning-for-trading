//! Feature engineering for trading models.
//!
//! Computes technical indicators and prepares features for ML models.

use super::{OhlcvData, TradingDataset};
use anyhow::Result;
use ndarray::{Array1, Array2};

/// Collection of trading features
#[derive(Debug, Clone)]
pub struct TradingFeatures {
    pub returns: Vec<f64>,
    pub log_returns: Vec<f64>,
    pub volatility: Vec<f64>,
    pub rsi: Vec<f64>,
    pub macd: Vec<f64>,
    pub macd_signal: Vec<f64>,
    pub bollinger_upper: Vec<f64>,
    pub bollinger_lower: Vec<f64>,
    pub atr: Vec<f64>,
    pub volume_ma: Vec<f64>,
}

impl TradingFeatures {
    /// Convert to ndarray matrix
    pub fn to_array(&self) -> Array2<f32> {
        let n = self.returns.len();
        let mut arr = Array2::<f32>::zeros((n, 10));

        for i in 0..n {
            arr[[i, 0]] = self.returns[i] as f32;
            arr[[i, 1]] = self.log_returns[i] as f32;
            arr[[i, 2]] = self.volatility[i] as f32;
            arr[[i, 3]] = self.rsi[i] as f32;
            arr[[i, 4]] = self.macd[i] as f32;
            arr[[i, 5]] = self.macd_signal[i] as f32;
            arr[[i, 6]] = self.bollinger_upper[i] as f32;
            arr[[i, 7]] = self.bollinger_lower[i] as f32;
            arr[[i, 8]] = self.atr[i] as f32;
            arr[[i, 9]] = self.volume_ma[i] as f32;
        }

        arr
    }
}

/// Calculate features from OHLCV data
pub fn calculate_features(data: &[OhlcvData]) -> TradingFeatures {
    let n = data.len();

    // Returns
    let mut returns = vec![0.0; n];
    let mut log_returns = vec![0.0; n];
    for i in 1..n {
        let ret = (data[i].close - data[i - 1].close) / data[i - 1].close;
        returns[i] = ret;
        log_returns[i] = (data[i].close / data[i - 1].close).ln();
    }

    // Volatility (20-period rolling std)
    let volatility = rolling_std(&returns, 20);

    // RSI (14-period)
    let rsi = calculate_rsi(&returns, 14);

    // MACD (12, 26, 9)
    let closes: Vec<f64> = data.iter().map(|d| d.close).collect();
    let (macd, macd_signal) = calculate_macd(&closes, 12, 26, 9);

    // Bollinger Bands (20-period, 2 std)
    let (bollinger_upper, bollinger_lower) = calculate_bollinger(&closes, 20, 2.0);

    // ATR (14-period)
    let atr = calculate_atr(data, 14);

    // Volume MA (20-period)
    let volumes: Vec<f64> = data.iter().map(|d| d.volume).collect();
    let volume_ma = simple_moving_average(&volumes, 20);

    TradingFeatures {
        returns,
        log_returns,
        volatility,
        rsi,
        macd,
        macd_signal,
        bollinger_upper,
        bollinger_lower,
        atr,
        volume_ma,
    }
}

/// Prepare features for Flash Attention model
pub fn prepare_features(
    data: &[OhlcvData],
    horizon: usize,
) -> Result<TradingDataset> {
    let n = data.len();
    if n < horizon + 50 {
        anyhow::bail!("Not enough data points");
    }

    let features = calculate_features(data);
    let feature_array = features.to_array();

    // Target: future returns
    let mut targets = Array1::<f32>::zeros(n);
    for i in 0..(n - horizon) {
        let future_ret = (data[i + horizon].close - data[i].close) / data[i].close;
        targets[i] = future_ret as f32;
    }

    // Remove NaN values from the beginning (due to rolling calculations)
    let start_idx = 50; // Skip first 50 to avoid NaN from indicators
    let end_idx = n - horizon;

    Ok(TradingDataset {
        features: feature_array.slice(ndarray::s![start_idx..end_idx, ..]).to_owned(),
        targets: targets.slice(ndarray::s![start_idx..end_idx]).to_owned(),
        timestamps: data[start_idx..end_idx].iter().map(|d| d.timestamp).collect(),
        prices: data[start_idx..end_idx].iter().map(|d| d.close).collect(),
    })
}

// Helper functions for technical indicators

fn simple_moving_average(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        if i < period - 1 {
            result[i] = data[..=i].iter().sum::<f64>() / (i + 1) as f64;
        } else {
            result[i] = data[(i + 1 - period)..=i].iter().sum::<f64>() / period as f64;
        }
    }

    result
}

fn exponential_moving_average(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    let alpha = 2.0 / (period + 1) as f64;

    result[0] = data[0];
    for i in 1..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = if i < period - 1 { 0 } else { i + 1 - period };
        let window = &data[start..=i];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
        result[i] = variance.sqrt();
    }

    result
}

fn calculate_rsi(returns: &[f64], period: usize) -> Vec<f64> {
    let n = returns.len();
    let mut result = vec![50.0; n]; // Default to neutral

    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 0..n {
        if returns[i] > 0.0 {
            gains[i] = returns[i];
        } else {
            losses[i] = -returns[i];
        }
    }

    let avg_gains = exponential_moving_average(&gains, period);
    let avg_losses = exponential_moving_average(&losses, period);

    for i in period..n {
        if avg_losses[i] > 0.0 {
            let rs = avg_gains[i] / avg_losses[i];
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    result
}

fn calculate_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let fast_ema = exponential_moving_average(prices, fast_period);
    let slow_ema = exponential_moving_average(prices, slow_period);

    let macd: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal = exponential_moving_average(&macd, signal_period);

    (macd, signal)
}

fn calculate_bollinger(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>) {
    let sma = simple_moving_average(prices, period);
    let std = rolling_std(prices, period);

    let upper: Vec<f64> = sma
        .iter()
        .zip(std.iter())
        .zip(prices.iter())
        .map(|((m, s), p)| (p - (m + num_std * s)) / p.max(0.0001)) // Normalized distance
        .collect();

    let lower: Vec<f64> = sma
        .iter()
        .zip(std.iter())
        .zip(prices.iter())
        .map(|((m, s), p)| (p - (m - num_std * s)) / p.max(0.0001)) // Normalized distance
        .collect();

    (upper, lower)
}

fn calculate_atr(data: &[OhlcvData], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut true_ranges = vec![0.0; n];

    for i in 0..n {
        let prev_close = if i > 0 { Some(data[i - 1].close) } else { None };
        true_ranges[i] = data[i].true_range(prev_close);
    }

    // Normalize by price
    let normalized: Vec<f64> = true_ranges
        .iter()
        .zip(data.iter())
        .map(|(tr, d)| tr / d.close.max(0.0001))
        .collect();

    exponential_moving_average(&normalized, period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_data(n: usize) -> Vec<OhlcvData> {
        let mut data = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = (i as f64 * 0.1).sin() * 5.0;
            price += change;

            data.push(OhlcvData {
                timestamp: Utc::now(),
                open: price - 1.0,
                high: price + 2.0,
                low: price - 2.0,
                close: price,
                volume: 1000.0 * (1.0 + (i as f64 * 0.05).cos()),
            });
        }

        data
    }

    #[test]
    fn test_calculate_features() {
        let data = create_test_data(100);
        let features = calculate_features(&data);

        assert_eq!(features.returns.len(), 100);
        assert_eq!(features.rsi.len(), 100);
        assert_eq!(features.macd.len(), 100);
    }

    #[test]
    fn test_prepare_features() {
        let data = create_test_data(200);
        let dataset = prepare_features(&data, 10).unwrap();

        assert!(dataset.features.nrows() > 0);
        assert_eq!(dataset.features.ncols(), 10);
    }

    #[test]
    fn test_simple_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = simple_moving_average(&data, 3);

        assert!((sma[2] - 2.0).abs() < 0.001);
        assert!((sma[4] - 4.0).abs() < 0.001);
    }
}
