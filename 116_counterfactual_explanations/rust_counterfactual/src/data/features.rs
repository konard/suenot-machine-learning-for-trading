//! Feature engineering utilities.

use crate::data::Candle;

/// Compute RSI (Relative Strength Index).
///
/// # Arguments
///
/// * `prices` - Slice of prices
/// * `period` - RSI period
///
/// # Returns
///
/// Vector of RSI values (NaN for initial values)
pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi = vec![f64::NAN; n];

    if n < period + 1 {
        return rsi;
    }

    // Calculate price changes
    let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    // Calculate initial averages
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 0..period {
        if changes[i] > 0.0 {
            avg_gain += changes[i];
        } else {
            avg_loss -= changes[i];
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    // First RSI value
    if avg_loss > 0.0 {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    } else {
        rsi[period] = 100.0;
    }

    // Calculate subsequent RSI values using smoothed averages
    for i in period..changes.len() {
        let change = changes[i];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            rsi[i + 1] = 100.0;
        }
    }

    rsi
}

/// Compute MACD (Moving Average Convergence Divergence).
///
/// # Arguments
///
/// * `prices` - Slice of prices
/// * `fast_period` - Fast EMA period
/// * `slow_period` - Slow EMA period
/// * `signal_period` - Signal line period
///
/// # Returns
///
/// Tuple of (MACD line, signal line, histogram)
pub fn compute_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();

    let ema_fast = compute_ema(prices, fast_period);
    let ema_slow = compute_ema(prices, slow_period);

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = compute_ema(&macd_line, signal_period);

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    (macd_line, signal_line, histogram)
}

/// Compute EMA (Exponential Moving Average).
fn compute_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut ema = vec![f64::NAN; n];

    if n < period {
        return ema;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Initial SMA
    let sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
    ema[period - 1] = sma;

    // Calculate EMA
    for i in period..n {
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    ema
}

/// Prepare features from candle data.
///
/// # Arguments
///
/// * `candles` - Slice of candle data
/// * `lookback` - Lookback period for indicators
///
/// # Returns
///
/// Tuple of (features matrix, feature names)
pub fn prepare_features(candles: &[Candle], lookback: usize) -> (Vec<Vec<f64>>, Vec<String>) {
    let n = candles.len();
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // Calculate indicators
    let rsi = compute_rsi(&prices, 14);
    let (macd, macd_signal, _) = compute_macd(&prices, 12, 26, 9);

    // Calculate returns
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Volume SMA
    let volume_sma = compute_sma(&volumes, lookback);

    // Price SMA for trend
    let price_sma = compute_sma(&prices, lookback);

    // Volatility (rolling std of returns)
    let volatility = compute_rolling_std(&returns, lookback);

    // Combine features
    let mut features = Vec::new();
    let start = lookback.max(26); // Start after all indicators are valid

    for i in start..n {
        if rsi[i].is_nan() || macd[i].is_nan() {
            continue;
        }

        let return_val = if i > 0 {
            (prices[i] - prices[i - 1]) / prices[i - 1]
        } else {
            0.0
        };

        let vol_ratio = if volume_sma[i] > 0.0 {
            volumes[i] / volume_sma[i] - 1.0
        } else {
            0.0
        };

        let trend = if price_sma[i] > 0.0 {
            (prices[i] - price_sma[i]) / price_sma[i]
        } else {
            0.0
        };

        let vol = if i > 0 && !volatility[i - 1].is_nan() {
            volatility[i - 1]
        } else {
            0.02
        };

        features.push(vec![
            (rsi[i] - 50.0) / 50.0,        // Normalized RSI
            macd[i] / prices[i],            // Normalized MACD
            vol_ratio,                       // Volume ratio
            vol,                             // Volatility
            trend,                           // Trend strength
            return_val,                      // Returns
        ]);
    }

    let feature_names = vec![
        "rsi_normalized".to_string(),
        "macd".to_string(),
        "volume_ratio".to_string(),
        "volatility".to_string(),
        "trend_strength".to_string(),
        "returns".to_string(),
    ];

    (features, feature_names)
}

/// Compute Simple Moving Average.
fn compute_sma(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut sma = vec![f64::NAN; n];

    if n < period {
        return sma;
    }

    for i in (period - 1)..n {
        let sum: f64 = values[(i + 1 - period)..=i].iter().sum();
        sma[i] = sum / period as f64;
    }

    sma
}

/// Compute rolling standard deviation.
fn compute_rolling_std(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut std = vec![f64::NAN; n];

    if n < period {
        return std;
    }

    for i in (period - 1)..n {
        let window = &values[(i + 1 - period)..=i];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        std[i] = variance.sqrt();
    }

    std
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rsi() {
        let prices = vec![44.0, 44.5, 43.5, 44.0, 44.5, 45.0, 45.5, 44.5, 44.0, 43.5, 44.0, 45.0, 46.0, 47.0, 46.0];
        let rsi = compute_rsi(&prices, 14);
        assert!(!rsi[14].is_nan());
        assert!(rsi[14] >= 0.0 && rsi[14] <= 100.0);
    }

    #[test]
    fn test_compute_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = compute_sma(&values, 3);
        assert!((sma[2] - 2.0).abs() < 0.001);
        assert!((sma[3] - 3.0).abs() < 0.001);
        assert!((sma[4] - 4.0).abs() < 0.001);
    }
}
