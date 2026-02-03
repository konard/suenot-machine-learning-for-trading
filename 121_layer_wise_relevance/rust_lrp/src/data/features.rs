//! Technical indicator and feature computation

use ndarray::{Array1, Array2, s};
use crate::api::Kline;

/// Feature names
pub const FEATURE_NAMES: &[&str] = &[
    "log_return", "return_5", "return_10",
    "volume_ratio", "volume_change",
    "volatility_5", "volatility_20",
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "bb_position",
    "atr_ratio",
    "ma_10", "ma_20",
];

/// Computed features for a single time step
#[derive(Debug, Clone)]
pub struct Features {
    pub log_return: f64,
    pub return_5: f64,
    pub return_10: f64,
    pub volume_ratio: f64,
    pub volume_change: f64,
    pub volatility_5: f64,
    pub volatility_20: f64,
    pub rsi_14: f64,
    pub rsi_7: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub bb_position: f64,
    pub atr_ratio: f64,
    pub ma_10: f64,
    pub ma_20: f64,
}

impl Features {
    /// Convert to array
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.log_return,
            self.return_5,
            self.return_10,
            self.volume_ratio,
            self.volume_change,
            self.volatility_5,
            self.volatility_20,
            self.rsi_14,
            self.rsi_7,
            self.macd,
            self.macd_signal,
            self.macd_hist,
            self.bb_position,
            self.atr_ratio,
            self.ma_10,
            self.ma_20,
        ])
    }

    /// Number of features
    pub const fn count() -> usize {
        16
    }
}

/// Compute RSI (Relative Strength Index)
pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let mut rsi = vec![50.0; prices.len()];

    if prices.len() < period + 1 {
        return rsi;
    }

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // Initial average
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss -= change;
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    if avg_loss > 0.0 {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Subsequent values using smoothed average
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            rsi[i] = 100.0;
        }
    }

    rsi
}

/// Compute MACD (Moving Average Convergence Divergence)
pub fn compute_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = ema(prices, fast_period);
    let slow_ema = ema(prices, slow_period);

    let macd_line: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = ema(&macd_line, signal_period);

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    (macd_line, signal_line, histogram)
}

/// Compute Exponential Moving Average
fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];

    if data.is_empty() || period == 0 {
        return result;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Start with SMA
    let sma: f64 = data.iter().take(period).sum::<f64>() / period as f64;
    result[period.saturating_sub(1)] = sma;

    // Calculate EMA
    for i in period..data.len() {
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
    }

    result
}

/// Compute Simple Moving Average
fn sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];

    for i in (period - 1)..data.len() {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        result[i] = sum / period as f64;
    }

    result
}

/// Compute rolling standard deviation
fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];

    for i in (period - 1)..data.len() {
        let slice = &data[(i + 1 - period)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / period as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Compute all features from klines
pub fn compute_features(klines: &[Kline]) -> Vec<Features> {
    let n = klines.len();
    if n < 30 {
        return vec![];
    }

    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    // Log returns
    let mut log_returns = vec![0.0; n];
    for i in 1..n {
        log_returns[i] = (closes[i] / closes[i - 1]).ln();
    }

    // Period returns
    let mut return_5 = vec![0.0; n];
    let mut return_10 = vec![0.0; n];
    for i in 5..n {
        return_5[i] = (closes[i] - closes[i - 5]) / closes[i - 5];
    }
    for i in 10..n {
        return_10[i] = (closes[i] - closes[i - 10]) / closes[i - 10];
    }

    // Volume features
    let volume_ma_20 = sma(&volumes, 20);
    let mut volume_ratio = vec![1.0; n];
    let mut volume_change = vec![0.0; n];
    for i in 20..n {
        volume_ratio[i] = volumes[i] / (volume_ma_20[i] + 1e-10);
    }
    for i in 1..n {
        volume_change[i] = (volumes[i] - volumes[i - 1]) / (volumes[i - 1] + 1e-10);
    }

    // Volatility
    let volatility_5 = rolling_std(&log_returns, 5);
    let volatility_20 = rolling_std(&log_returns, 20);

    // RSI
    let rsi_14 = compute_rsi(&closes, 14);
    let rsi_7 = compute_rsi(&closes, 7);

    // MACD
    let (macd, macd_signal, macd_hist) = compute_macd(&closes, 12, 26, 9);

    // Bollinger Bands position
    let ma_20 = sma(&closes, 20);
    let std_20 = rolling_std(&closes, 20);
    let mut bb_position = vec![0.5; n];
    for i in 20..n {
        let upper = ma_20[i] + 2.0 * std_20[i];
        let lower = ma_20[i] - 2.0 * std_20[i];
        let range = upper - lower;
        if range > 0.0 {
            bb_position[i] = ((closes[i] - lower) / range).clamp(0.0, 1.0);
        }
    }

    // ATR
    let mut tr = vec![0.0; n];
    for i in 1..n {
        tr[i] = klines[i].true_range(closes[i - 1]);
    }
    let atr_14 = sma(&tr, 14);
    let mut atr_ratio = vec![0.0; n];
    for i in 14..n {
        atr_ratio[i] = atr_14[i] / (closes[i] + 1e-10);
    }

    // Moving average ratios
    let ma_10 = sma(&closes, 10);
    let mut ma_10_ratio = vec![1.0; n];
    let mut ma_20_ratio = vec![1.0; n];
    for i in 10..n {
        ma_10_ratio[i] = ma_10[i] / (closes[i] + 1e-10);
    }
    for i in 20..n {
        ma_20_ratio[i] = ma_20[i] / (closes[i] + 1e-10);
    }

    // Combine into Features
    let mut features = Vec::with_capacity(n);
    for i in 0..n {
        features.push(Features {
            log_return: log_returns[i],
            return_5: return_5[i],
            return_10: return_10[i],
            volume_ratio: volume_ratio[i],
            volume_change: volume_change[i],
            volatility_5: volatility_5[i],
            volatility_20: volatility_20[i],
            rsi_14: rsi_14[i] / 100.0,  // Normalize to 0-1
            rsi_7: rsi_7[i] / 100.0,
            macd: macd[i] / (closes[i] + 1e-10),  // Normalize
            macd_signal: macd_signal[i] / (closes[i] + 1e-10),
            macd_hist: macd_hist[i] / (closes[i] + 1e-10),
            bb_position: bb_position[i],
            atr_ratio: atr_ratio[i],
            ma_10: ma_10_ratio[i],
            ma_20: ma_20_ratio[i],
        });
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.5, 44.25, 44.75, 45.0, 44.5, 44.75, 45.25, 45.0, 44.75,
                         45.0, 45.5, 45.25, 45.75, 46.0, 45.5];
        let rsi = compute_rsi(&prices, 14);
        assert!(rsi[14] > 0.0 && rsi[14] < 100.0);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema(&data, 3);
        assert!(result[9] > result[2]);
    }
}
