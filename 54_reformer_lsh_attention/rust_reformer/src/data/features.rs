//! Feature engineering for cryptocurrency data
//!
//! Computes technical indicators and features for model input.

use crate::api::Kline;
use ndarray::{Array1, Array2};

/// Feature computation results
#[derive(Debug, Clone)]
pub struct Features {
    /// Feature matrix [seq_len, n_features]
    pub data: Array2<f64>,
    /// Feature names
    pub names: Vec<String>,
}

impl Features {
    /// Number of features
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Sequence length
    pub fn seq_len(&self) -> usize {
        self.data.nrows()
    }

    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<Array1<f64>> {
        let idx = self.names.iter().position(|n| n == name)?;
        Some(self.data.column(idx).to_owned())
    }
}

/// Compute log returns
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![0.0; prices.len()];
    }

    let mut returns = vec![0.0];
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            returns.push((prices[i] / prices[i - 1]).ln());
        } else {
            returns.push(0.0);
        }
    }
    returns
}

/// Compute simple returns
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![0.0; prices.len()];
    }

    let mut returns = vec![0.0];
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        } else {
            returns.push(0.0);
        }
    }
    returns
}

/// Compute rolling volatility (standard deviation)
pub fn volatility(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut vol = vec![0.0; n];

    for i in 0..n {
        if i < window - 1 {
            // Not enough data, use available
            let slice = &returns[0..=i];
            vol[i] = std_dev(slice);
        } else {
            let slice = &returns[i + 1 - window..=i];
            vol[i] = std_dev(slice);
        }
    }

    vol
}

/// Compute RSI (Relative Strength Index)
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi_values = vec![50.0; n]; // Default neutral

    if n < period + 1 {
        return rsi_values;
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

    // Calculate initial averages
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    // Calculate RSI
    for i in period..n {
        if avg_loss == 0.0 {
            rsi_values[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Update averages using smoothing
        if i < n - 1 {
            avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        }
    }

    rsi_values
}

/// Compute MACD (Moving Average Convergence Divergence)
pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();

    let ema_fast = ema(prices, fast);
    let ema_slow = ema(prices, slow);

    // MACD line
    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    // Signal line
    let signal_line = ema(&macd_line, signal);

    // Histogram
    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    (macd_line, signal_line, histogram)
}

/// Compute Exponential Moving Average
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut ema_values = vec![0.0; n];

    if n == 0 || period == 0 {
        return ema_values;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Initialize with SMA
    let sma: f64 = data.iter().take(period).sum::<f64>() / period.min(n) as f64;
    ema_values[0] = if n > 0 { data[0] } else { 0.0 };

    for i in 1..n {
        if i < period {
            ema_values[i] = data[..=i].iter().sum::<f64>() / (i + 1) as f64;
        } else if i == period {
            ema_values[i] = sma;
        } else {
            ema_values[i] = (data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1];
        }
    }

    ema_values
}

/// Compute Simple Moving Average
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut sma_values = vec![0.0; n];

    for i in 0..n {
        let start = if i >= period { i + 1 - period } else { 0 };
        let sum: f64 = data[start..=i].iter().sum();
        sma_values[i] = sum / (i + 1 - start) as f64;
    }

    sma_values
}

/// Compute Bollinger Bands
pub fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let middle = sma(prices, period);

    let mut upper = vec![0.0; n];
    let mut lower = vec![0.0; n];

    for i in 0..n {
        let start = if i >= period { i + 1 - period } else { 0 };
        let slice = &prices[start..=i];
        let std = std_dev(slice);

        upper[i] = middle[i] + num_std * std;
        lower[i] = middle[i] - num_std * std;
    }

    (upper, middle, lower)
}

/// Compute ATR (Average True Range)
pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
    let n = klines.len();
    let mut tr = vec![0.0; n];

    // Calculate True Range
    for i in 0..n {
        let high_low = klines[i].high - klines[i].low;
        if i == 0 {
            tr[i] = high_low;
        } else {
            let high_close = (klines[i].high - klines[i - 1].close).abs();
            let low_close = (klines[i].low - klines[i - 1].close).abs();
            tr[i] = high_low.max(high_close).max(low_close);
        }
    }

    // Calculate ATR using EMA
    ema(&tr, period)
}

/// Compute volume profile features
pub fn volume_features(klines: &[Kline]) -> (Vec<f64>, Vec<f64>) {
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    // Volume ratio (current / rolling average)
    let vol_ma = sma(&volumes, 20);
    let vol_ratio: Vec<f64> = volumes
        .iter()
        .zip(vol_ma.iter())
        .map(|(v, m)| if *m > 0.0 { v / m } else { 1.0 })
        .collect();

    // Volume change
    let vol_change = simple_returns(&volumes);

    (vol_ratio, vol_change)
}

/// Compute OBV (On-Balance Volume)
pub fn obv(klines: &[Kline]) -> Vec<f64> {
    let n = klines.len();
    let mut obv_values = vec![0.0; n];

    if n > 0 {
        obv_values[0] = klines[0].volume;
    }

    for i in 1..n {
        if klines[i].close > klines[i - 1].close {
            obv_values[i] = obv_values[i - 1] + klines[i].volume;
        } else if klines[i].close < klines[i - 1].close {
            obv_values[i] = obv_values[i - 1] - klines[i].volume;
        } else {
            obv_values[i] = obv_values[i - 1];
        }
    }

    obv_values
}

/// Compute all features from klines
pub fn compute_all_features(klines: &[Kline]) -> Features {
    let n = klines.len();

    // Extract price data
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let opens: Vec<f64> = klines.iter().map(|k| k.open).collect();
    let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
    let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();

    // Returns
    let log_ret = log_returns(&closes);
    let simple_ret = simple_returns(&closes);

    // Technical indicators
    let vol = volatility(&log_ret, 20);
    let rsi_14 = rsi(&closes, 14);
    let (macd_line, signal_line, macd_hist) = macd(&closes, 12, 26, 9);
    let (bb_upper, bb_middle, bb_lower) = bollinger_bands(&closes, 20, 2.0);
    let atr_14 = atr(klines, 14);
    let (vol_ratio, vol_change) = volume_features(klines);
    let obv_values = obv(klines);

    // Price position features
    let bb_position: Vec<f64> = closes
        .iter()
        .zip(bb_upper.iter().zip(bb_lower.iter()))
        .map(|(c, (u, l))| {
            if (u - l).abs() > 1e-10 {
                (c - l) / (u - l)
            } else {
                0.5
            }
        })
        .collect();

    // Range features
    let hl_range: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .map(|(h, l)| h - l)
        .collect();

    // Build feature matrix
    let feature_vecs = vec![
        log_ret,
        vol,
        rsi_14,
        macd_line,
        macd_hist,
        bb_position,
        atr_14,
        vol_ratio,
        vol_change,
        hl_range,
    ];

    let n_features = feature_vecs.len();
    let mut data = Array2::zeros((n, n_features));

    for (col, vec) in feature_vecs.iter().enumerate() {
        for (row, &val) in vec.iter().enumerate() {
            data[[row, col]] = val;
        }
    }

    // Normalize features
    let data = normalize_features(data);

    Features {
        data,
        names: vec![
            "log_return".to_string(),
            "volatility".to_string(),
            "rsi".to_string(),
            "macd".to_string(),
            "macd_hist".to_string(),
            "bb_position".to_string(),
            "atr".to_string(),
            "volume_ratio".to_string(),
            "volume_change".to_string(),
            "range".to_string(),
        ],
    }
}

/// Normalize features to zero mean and unit variance
fn normalize_features(mut data: Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = data.dim();

    for col in 0..n_cols {
        let column = data.column(col);
        let mean = column.mean().unwrap_or(0.0);
        let std = std_dev_array(&column);

        if std > 1e-10 {
            for row in 0..n_rows {
                data[[row, col]] = (data[[row, col]] - mean) / std;
            }
        } else {
            // Zero variance, just center
            for row in 0..n_rows {
                data[[row, col]] = data[[row, col]] - mean;
            }
        }
    }

    // Clip extreme values
    data.mapv_inplace(|x| x.max(-5.0).min(5.0));

    data
}

/// Standard deviation helper
fn std_dev(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

/// Standard deviation for ndarray
fn std_dev_array(data: &ndarray::ArrayView1<f64>) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    let mean = data.mean().unwrap_or(0.0);
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert_eq!(returns[0], 0.0);
        assert!((returns[1] - 0.0488).abs() < 0.001);
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert_eq!(result.len(), 5);
        assert!((result[2] - 2.0).abs() < 0.001); // (1+2+3)/3
        assert!((result[4] - 4.0).abs() < 0.001); // (3+4+5)/3
    }

    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 45.5, 45.0];
        let rsi_values = rsi(&prices, 14);

        assert_eq!(rsi_values.len(), prices.len());
        // RSI should be between 0 and 100
        for val in &rsi_values {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_bollinger_bands() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let (upper, middle, lower) = bollinger_bands(&prices, 5, 2.0);

        assert_eq!(upper.len(), 20);
        assert_eq!(middle.len(), 20);
        assert_eq!(lower.len(), 20);

        // Upper should be above middle, lower below
        for i in 0..20 {
            assert!(upper[i] >= middle[i]);
            assert!(lower[i] <= middle[i]);
        }
    }

    #[test]
    fn test_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.015];
        let vol = volatility(&returns, 3);

        assert_eq!(vol.len(), 6);
        for val in &vol {
            assert!(*val >= 0.0);
        }
    }
}
