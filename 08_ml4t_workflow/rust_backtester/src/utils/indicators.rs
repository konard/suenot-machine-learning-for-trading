//! Technical indicators.

use crate::models::Candle;

/// Calculate Simple Moving Average.
pub fn sma(data: &[f64], period: usize) -> Vec<Option<f64>> {
    if period == 0 || data.is_empty() {
        return vec![None; data.len()];
    }

    data.iter()
        .enumerate()
        .map(|(i, _)| {
            if i + 1 < period {
                None
            } else {
                let sum: f64 = data[i + 1 - period..=i].iter().sum();
                Some(sum / period as f64)
            }
        })
        .collect()
}

/// Calculate Exponential Moving Average.
pub fn ema(data: &[f64], period: usize) -> Vec<Option<f64>> {
    if period == 0 || data.is_empty() {
        return vec![None; data.len()];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = vec![None; data.len()];

    // First EMA value is the SMA
    if data.len() >= period {
        let first_sma: f64 = data[0..period].iter().sum::<f64>() / period as f64;
        result[period - 1] = Some(first_sma);

        // Calculate subsequent EMA values
        let mut prev_ema = first_sma;
        for i in period..data.len() {
            let ema = (data[i] - prev_ema) * multiplier + prev_ema;
            result[i] = Some(ema);
            prev_ema = ema;
        }
    }

    result
}

/// Calculate Relative Strength Index (RSI).
pub fn rsi(data: &[f64], period: usize) -> Vec<Option<f64>> {
    if period == 0 || data.len() < 2 {
        return vec![None; data.len()];
    }

    let mut result = vec![None; data.len()];
    let mut gains = Vec::with_capacity(data.len() - 1);
    let mut losses = Vec::with_capacity(data.len() - 1);

    // Calculate gains and losses
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    if gains.len() < period {
        return result;
    }

    // First RSI uses SMA
    let mut avg_gain: f64 = gains[0..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[0..period].iter().sum::<f64>() / period as f64;

    result[period] = Some(calculate_rsi_value(avg_gain, avg_loss));

    // Subsequent RSI uses smoothed average
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        result[i + 1] = Some(calculate_rsi_value(avg_gain, avg_loss));
    }

    result
}

fn calculate_rsi_value(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

/// Calculate MACD (Moving Average Convergence Divergence).
pub struct MacdResult {
    pub macd_line: Vec<Option<f64>>,
    pub signal_line: Vec<Option<f64>>,
    pub histogram: Vec<Option<f64>>,
}

pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> MacdResult {
    let ema_fast = ema(data, fast);
    let ema_slow = ema(data, slow);

    let macd_line: Vec<Option<f64>> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| match (f, s) {
            (Some(f), Some(s)) => Some(f - s),
            _ => None,
        })
        .collect();

    // Extract MACD values for signal line calculation
    let macd_values: Vec<f64> = macd_line
        .iter()
        .filter_map(|x| *x)
        .collect();

    let signal_ema = ema(&macd_values, signal);

    // Map signal EMA back to full length
    let mut signal_line = vec![None; data.len()];
    let mut signal_idx = 0;
    for (i, m) in macd_line.iter().enumerate() {
        if m.is_some() {
            if signal_idx < signal_ema.len() {
                signal_line[i] = signal_ema[signal_idx];
            }
            signal_idx += 1;
        }
    }

    let histogram: Vec<Option<f64>> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| match (m, s) {
            (Some(m), Some(s)) => Some(m - s),
            _ => None,
        })
        .collect();

    MacdResult {
        macd_line,
        signal_line,
        histogram,
    }
}

/// Calculate Bollinger Bands.
pub struct BollingerBands {
    pub upper: Vec<Option<f64>>,
    pub middle: Vec<Option<f64>>,
    pub lower: Vec<Option<f64>>,
}

pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> BollingerBands {
    let middle = sma(data, period);

    let mut upper = vec![None; data.len()];
    let mut lower = vec![None; data.len()];

    for i in period - 1..data.len() {
        if let Some(sma_val) = middle[i] {
            let slice = &data[i + 1 - period..=i];
            let variance: f64 = slice.iter().map(|x| (x - sma_val).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();
            upper[i] = Some(sma_val + std_dev * std);
            lower[i] = Some(sma_val - std_dev * std);
        }
    }

    BollingerBands { upper, middle, lower }
}

/// Calculate Average True Range (ATR).
pub fn atr(candles: &[Candle], period: usize) -> Vec<Option<f64>> {
    if candles.len() < 2 || period == 0 {
        return vec![None; candles.len()];
    }

    let mut true_ranges = Vec::with_capacity(candles.len());
    true_ranges.push(candles[0].high - candles[0].low);

    for i in 1..candles.len() {
        let high = candles[i].high;
        let low = candles[i].low;
        let prev_close = candles[i - 1].close;

        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        true_ranges.push(tr);
    }

    // Use EMA for smoothed ATR
    ema(&true_ranges, period)
}

/// Calculate returns from prices.
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Calculate log returns from prices.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert!(result[0].is_none());
        assert!(result[1].is_none());
        assert!((result[2].unwrap() - 2.0).abs() < 0.001);
        assert!((result[3].unwrap() - 3.0).abs() < 0.001);
        assert!((result[4].unwrap() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5, 44.0, 43.5, 43.0, 43.5, 44.0];
        let result = rsi(&data, 14);
        assert!(result[14].is_some());
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 105.0, 102.0];
        let ret = returns(&prices);
        assert!((ret[0] - 0.05).abs() < 0.001);
        assert!((ret[1] - (-0.0286)).abs() < 0.01);
    }
}
