//! Technical indicators for risk assessment
//!
//! Provides various indicators useful for detecting market stress

use crate::data::OHLCVSeries;

/// Calculate Simple Moving Average
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        result.push(sum / period as f64);
    }

    result
}

/// Calculate Exponential Moving Average
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());

    // First value is the same as SMA
    let first_sma: f64 = data.iter().take(period).sum::<f64>() / period as f64;

    for (i, &value) in data.iter().enumerate() {
        if i < period - 1 {
            result.push(f64::NAN);
        } else if i == period - 1 {
            result.push(first_sma);
        } else {
            let prev_ema = result[i - 1];
            let new_ema = (value - prev_ema) * multiplier + prev_ema;
            result.push(new_ema);
        }
    }

    result
}

/// Calculate rolling standard deviation
pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        result.push(variance.sqrt());
    }

    result
}

/// Calculate rolling correlation between two series
pub fn rolling_correlation(x: &[f64], y: &[f64], period: usize) -> Vec<f64> {
    let len = x.len().min(y.len());
    if len < period {
        return vec![f64::NAN; len];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..len {
        let wx = &x[(i + 1 - period)..=i];
        let wy = &y[(i + 1 - period)..=i];

        let mean_x: f64 = wx.iter().sum::<f64>() / period as f64;
        let mean_y: f64 = wy.iter().sum::<f64>() / period as f64;

        let cov: f64 = wx
            .iter()
            .zip(wy)
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / period as f64;

        let var_x: f64 = wx.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / period as f64;
        let var_y: f64 = wy.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / period as f64;

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        if std_x < 1e-10 || std_y < 1e-10 {
            result.push(0.0);
        } else {
            result.push(cov / (std_x * std_y));
        }
    }

    result
}

/// Calculate Average True Range (ATR)
pub fn atr(data: &OHLCVSeries, period: usize) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }

    // Calculate True Range
    let mut tr = vec![data.data[0].high - data.data[0].low];

    for i in 1..data.len() {
        let curr = &data.data[i];
        let prev_close = data.data[i - 1].close;

        let tr1 = curr.high - curr.low;
        let tr2 = (curr.high - prev_close).abs();
        let tr3 = (curr.low - prev_close).abs();

        tr.push(tr1.max(tr2).max(tr3));
    }

    // Calculate ATR using EMA
    ema(&tr, period)
}

/// Calculate Bollinger Bands
pub fn bollinger_bands(data: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = sma(data, period);
    let std = rolling_std(data, period);

    let upper: Vec<f64> = middle
        .iter()
        .zip(&std)
        .map(|(m, s)| m + num_std * s)
        .collect();

    let lower: Vec<f64> = middle
        .iter()
        .zip(&std)
        .map(|(m, s)| m - num_std * s)
        .collect();

    (upper, middle, lower)
}

/// Calculate Bollinger Band Width (volatility indicator)
pub fn bollinger_width(data: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let (upper, middle, lower) = bollinger_bands(data, period, num_std);

    upper
        .iter()
        .zip(&middle)
        .zip(&lower)
        .map(|((u, m), l)| {
            if m.abs() < 1e-10 {
                0.0
            } else {
                (u - l) / m
            }
        })
        .collect()
}

/// Calculate rolling percentile rank
pub fn rolling_percentile_rank(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![0.5; data.len()];
    }

    let mut result = vec![0.5; period - 1];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let current = data[i];

        let count_below = window.iter().filter(|&&x| x < current).count();
        result.push(count_below as f64 / period as f64);
    }

    result
}

/// Calculate rolling maximum drawdown
pub fn rolling_max_drawdown(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![0.0; data.len()];
    }

    let mut result = vec![0.0; period - 1];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];

        let mut peak = window[0];
        let mut max_dd = 0.0;

        for &value in window {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        result.push(max_dd);
    }

    result
}

/// Calculate Rate of Change (ROC)
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() <= period {
        return vec![0.0; data.len()];
    }

    let mut result = vec![0.0; period];

    for i in period..data.len() {
        let prev = data[i - period];
        if prev.abs() > 1e-10 {
            result.push((data[i] - prev) / prev * 100.0);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Calculate RSI (Relative Strength Index)
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![50.0; data.len()];
    }

    // Calculate price changes
    let changes: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();

    let gains: Vec<f64> = changes.iter().map(|&c| if c > 0.0 { c } else { 0.0 }).collect();
    let losses: Vec<f64> = changes
        .iter()
        .map(|&c| if c < 0.0 { -c } else { 0.0 })
        .collect();

    // Use EMA for smoothing
    let avg_gain = ema(&gains, period);
    let avg_loss = ema(&losses, period);

    let mut result = vec![50.0];

    for i in 0..changes.len() {
        let ag = avg_gain.get(i).copied().unwrap_or(0.0);
        let al = avg_loss.get(i).copied().unwrap_or(0.0);

        if al < 1e-10 {
            result.push(100.0);
        } else {
            let rs = ag / al;
            result.push(100.0 - 100.0 / (1.0 + rs));
        }
    }

    result
}

/// Calculate Volume-Weighted Average Price (VWAP) deviation
pub fn vwap_deviation(prices: &[f64], volumes: &[f64]) -> Vec<f64> {
    if prices.len() != volumes.len() || prices.is_empty() {
        return Vec::new();
    }

    let mut cumulative_pv = 0.0;
    let mut cumulative_v = 0.0;
    let mut result = Vec::with_capacity(prices.len());

    for (&price, &volume) in prices.iter().zip(volumes) {
        cumulative_pv += price * volume;
        cumulative_v += volume;

        if cumulative_v > 0.0 {
            let vwap = cumulative_pv / cumulative_v;
            result.push((price - vwap) / vwap * 100.0);
        } else {
            result.push(0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-6);
        assert!((result[3] - 3.0).abs() < 1e-6);
        assert!((result[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 1.0, 1.0, 1.0, 10.0]; // Last value is outlier
        let result = rolling_std(&data, 4);

        // Standard deviation should spike at the end
        assert!(result.last().unwrap() > &1.0);
    }

    #[test]
    fn test_rsi() {
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = rsi(&data, 14);

        // Uptrend should have high RSI
        assert!(result.last().unwrap() > &50.0);
    }
}
