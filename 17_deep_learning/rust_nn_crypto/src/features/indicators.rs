//! Technical Indicators
//!
//! Common technical analysis indicators for trading

use crate::data::OHLCVSeries;

/// Simple Moving Average
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

/// Exponential Moving Average
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = vec![data[0]];

    for i in 1..data.len() {
        let ema_val = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        result.push(ema_val);
    }

    result
}

/// Relative Strength Index
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![f64::NAN; data.len()];
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

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

    let mut result = vec![f64::NAN; period];

    // Initial average
    let avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
    result.push(100.0 - 100.0 / (1.0 + rs));

    // Smoothed RSI
    let mut prev_avg_gain = avg_gain;
    let mut prev_avg_loss = avg_loss;

    for i in period..gains.len() {
        let avg_gain = (prev_avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        let avg_loss = (prev_avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
        result.push(100.0 - 100.0 / (1.0 + rs));

        prev_avg_gain = avg_gain;
        prev_avg_loss = avg_loss;
    }

    result
}

/// Moving Average Convergence Divergence
pub struct MACD {
    pub macd_line: Vec<f64>,
    pub signal_line: Vec<f64>,
    pub histogram: Vec<f64>,
}

pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> MACD {
    let ema_fast = ema(data, fast);
    let ema_slow = ema(data, slow);

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(&f, &s)| f - s)
        .collect();

    let signal_line = ema(&macd_line, signal);

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(&m, &s)| m - s)
        .collect();

    MACD {
        macd_line,
        signal_line,
        histogram,
    }
}

/// Bollinger Bands
pub struct BollingerBands {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> BollingerBands {
    let middle = sma(data, period);

    let mut upper = Vec::with_capacity(data.len());
    let mut lower = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        if i < period - 1 {
            upper.push(f64::NAN);
            lower.push(f64::NAN);
        } else {
            let window = &data[(i + 1 - period)..=i];
            let mean = middle[i];
            let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper.push(mean + std_dev * std);
            lower.push(mean - std_dev * std);
        }
    }

    BollingerBands { upper, middle, lower }
}

/// Average True Range
pub fn atr(series: &OHLCVSeries, period: usize) -> Vec<f64> {
    if series.len() < 2 {
        return vec![f64::NAN; series.len()];
    }

    let mut tr = Vec::with_capacity(series.len());
    tr.push(series.data[0].high - series.data[0].low);

    for i in 1..series.len() {
        let high = series.data[i].high;
        let low = series.data[i].low;
        let prev_close = series.data[i - 1].close;

        let tr_val = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        tr.push(tr_val);
    }

    // Use EMA for smoothing
    ema(&tr, period)
}

/// Stochastic Oscillator
pub struct Stochastic {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

pub fn stochastic(series: &OHLCVSeries, k_period: usize, d_period: usize) -> Stochastic {
    if series.len() < k_period {
        return Stochastic {
            k: vec![f64::NAN; series.len()],
            d: vec![f64::NAN; series.len()],
        };
    }

    let mut k = vec![f64::NAN; k_period - 1];

    for i in (k_period - 1)..series.len() {
        let window = &series.data[(i + 1 - k_period)..=i];

        let highest = window.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest = window.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let close = series.data[i].close;

        let k_val = if highest != lowest {
            ((close - lowest) / (highest - lowest)) * 100.0
        } else {
            50.0
        };
        k.push(k_val);
    }

    let d = sma(&k, d_period);

    Stochastic { k, d }
}

/// On-Balance Volume
pub fn obv(series: &OHLCVSeries) -> Vec<f64> {
    if series.is_empty() {
        return vec![];
    }

    let mut result = vec![0.0];

    for i in 1..series.len() {
        let prev_obv = result[i - 1];
        let volume = series.data[i].volume;
        let close = series.data[i].close;
        let prev_close = series.data[i - 1].close;

        let obv_val = if close > prev_close {
            prev_obv + volume
        } else if close < prev_close {
            prev_obv - volume
        } else {
            prev_obv
        };
        result.push(obv_val);
    }

    result
}

/// Money Flow Index
pub fn mfi(series: &OHLCVSeries, period: usize) -> Vec<f64> {
    if series.len() < period + 1 {
        return vec![f64::NAN; series.len()];
    }

    // Calculate typical price and raw money flow
    let mut positive_flow = Vec::new();
    let mut negative_flow = Vec::new();

    for i in 1..series.len() {
        let tp = series.data[i].typical_price();
        let prev_tp = series.data[i - 1].typical_price();
        let mf = tp * series.data[i].volume;

        if tp > prev_tp {
            positive_flow.push(mf);
            negative_flow.push(0.0);
        } else {
            positive_flow.push(0.0);
            negative_flow.push(mf);
        }
    }

    let mut result = vec![f64::NAN; period];

    for i in period..positive_flow.len() + 1 {
        let pos_sum: f64 = positive_flow[(i - period)..i].iter().sum();
        let neg_sum: f64 = negative_flow[(i - period)..i].iter().sum();

        let mfi_val = if neg_sum > 0.0 {
            100.0 - 100.0 / (1.0 + pos_sum / neg_sum)
        } else {
            100.0
        };
        result.push(mfi_val);
    }

    result
}

/// Returns
pub fn returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![];
    }

    data.windows(2)
        .map(|w| if w[0] != 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 })
        .collect()
}

/// Log returns
pub fn log_returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![];
    }

    data.windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Rolling standard deviation
pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        result.push(variance.sqrt());
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
        assert_eq!(result.len(), 5);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);
        assert_eq!(result.len(), 5);
        assert!(result[4] > result[0]);
    }

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 44.5, 44.25, 44.0, 43.5, 44.0, 44.25, 44.5, 44.25];
        let result = rsi(&data, 14);
        assert_eq!(result.len(), 15);
        // RSI should be between 0 and 100
        for &val in result.iter().skip(14) {
            assert!(val >= 0.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 110.0, 121.0];
        let result = returns(&data);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.1).abs() < 1e-10);
        assert!((result[1] - 0.1).abs() < 1e-10);
    }
}
