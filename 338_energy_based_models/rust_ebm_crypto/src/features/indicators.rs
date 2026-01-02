//! Technical indicators for feature engineering

/// Compute simple moving average
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        result.push(sum / period as f64);
    }

    result
}

/// Compute exponential moving average
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; data.len()];

    // Initialize with first value
    result[0] = data[0];

    for i in 1..data.len() {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

/// Compute rolling standard deviation
pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period || period == 0 {
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

/// Compute returns (percentage change)
pub fn returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN];

    for i in 1..data.len() {
        if data[i - 1] != 0.0 {
            result.push((data[i] - data[i - 1]) / data[i - 1]);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Compute log returns
pub fn log_returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN];

    for i in 1..data.len() {
        if data[i - 1] > 0.0 && data[i] > 0.0 {
            result.push((data[i] / data[i - 1]).ln());
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Compute rolling skewness
pub fn rolling_skewness(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period || period < 3 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let n = period as f64;
        let mean: f64 = window.iter().sum::<f64>() / n;

        let m2: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = window.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;

        let std = m2.sqrt();
        if std > 1e-10 {
            result.push(m3 / std.powi(3));
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Compute rolling kurtosis
pub fn rolling_kurtosis(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period || period < 4 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let n = period as f64;
        let mean: f64 = window.iter().sum::<f64>() / n;

        let m2: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4: f64 = window.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

        let variance = m2;
        if variance > 1e-10 {
            // Excess kurtosis (normal = 0)
            result.push(m4 / variance.powi(2) - 3.0);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Compute Z-score
pub fn zscore(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let means = sma(data, period);
    let stds = rolling_std(data, period);

    data.iter()
        .zip(means.iter())
        .zip(stds.iter())
        .map(|((x, m), s)| {
            if s.is_nan() || m.is_nan() || *s < 1e-10 {
                f64::NAN
            } else {
                (x - m) / s
            }
        })
        .collect()
}

/// Compute RSI (Relative Strength Index)
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut gains = vec![0.0; data.len()];
    let mut losses = vec![0.0; data.len()];

    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    let avg_gains = ema(&gains, period);
    let avg_losses = ema(&losses, period);

    avg_gains
        .iter()
        .zip(avg_losses.iter())
        .map(|(g, l)| {
            if l.is_nan() || g.is_nan() || *l < 1e-10 {
                50.0 // Neutral
            } else {
                100.0 - 100.0 / (1.0 + g / l)
            }
        })
        .collect()
}

/// Compute Bollinger Bands position (-1 to 1)
pub fn bollinger_position(data: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let means = sma(data, period);
    let stds = rolling_std(data, period);

    data.iter()
        .zip(means.iter())
        .zip(stds.iter())
        .map(|((x, m), s)| {
            if s.is_nan() || m.is_nan() || *s < 1e-10 {
                0.0
            } else {
                let upper = m + num_std * s;
                let lower = m - num_std * s;
                let band_width = upper - lower;
                if band_width > 1e-10 {
                    (2.0 * (x - lower) / band_width) - 1.0
                } else {
                    0.0
                }
            }
        })
        .collect()
}

/// Compute Average True Range (ATR)
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() || high.len() < 2 {
        return vec![f64::NAN; high.len()];
    }

    let mut true_ranges = vec![high[0] - low[0]];

    for i in 1..high.len() {
        let tr = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
        true_ranges.push(tr);
    }

    ema(&true_ranges, period)
}

/// Compute momentum (rate of change)
pub fn momentum(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        if data[i - period] != 0.0 {
            result.push((data[i] - data[i - period]) / data[i - period]);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Compute volume-weighted average price (VWAP) ratio
pub fn vwap_ratio(close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    if close.len() != volume.len() || close.len() < period {
        return vec![f64::NAN; close.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..close.len() {
        let mut sum_pv = 0.0;
        let mut sum_v = 0.0;

        for j in (i + 1 - period)..=i {
            sum_pv += close[j] * volume[j];
            sum_v += volume[j];
        }

        let vwap = if sum_v > 0.0 { sum_pv / sum_v } else { close[i] };
        result.push(close[i] / vwap);
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
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 110.0, 99.0];
        let result = returns(&data);

        assert!(result[0].is_nan());
        assert!((result[1] - 0.1).abs() < 1e-10);
        assert!((result[2] - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = zscore(&data, 5);

        // Last value should be positive (above mean)
        assert!(result.last().unwrap() > &0.0);
    }

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.5, 43.5, 44.5, 45.0, 45.5, 46.0, 45.5, 46.5, 47.0];
        let result = rsi(&data, 5);

        // RSI should be between 0 and 100
        for val in result.iter().skip(5) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }
}
