//! Technical indicators for feature engineering
//!
//! Provides various technical indicators commonly used in trading

use crate::data::{rolling_mean, rolling_std, OHLCVSeries};

/// Calculate Simple Moving Average (SMA)
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    rolling_mean(data, period)
}

/// Calculate Exponential Moving Average (EMA)
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len());
    let multiplier = 2.0 / (period as f64 + 1.0);

    // First EMA is the SMA of the first 'period' values
    if data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let first_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;

    // Fill NaNs for the initial period
    for _ in 0..period - 1 {
        result.push(f64::NAN);
    }
    result.push(first_sma);

    // Calculate EMA for the rest
    let mut prev_ema = first_sma;
    for &value in &data[period..] {
        let current_ema = (value - prev_ema) * multiplier + prev_ema;
        result.push(current_ema);
        prev_ema = current_ema;
    }

    result
}

/// Calculate Relative Strength Index (RSI)
pub fn rsi(closes: &[f64], period: usize) -> Vec<f64> {
    if closes.len() < 2 || period == 0 {
        return vec![f64::NAN; closes.len()];
    }

    let mut gains = Vec::with_capacity(closes.len());
    let mut losses = Vec::with_capacity(closes.len());

    gains.push(0.0);
    losses.push(0.0);

    for i in 1..closes.len() {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    let avg_gains = ema(&gains, period);
    let avg_losses = ema(&losses, period);

    avg_gains
        .iter()
        .zip(avg_losses.iter())
        .map(|(&gain, &loss)| {
            if gain.is_nan() || loss.is_nan() {
                f64::NAN
            } else if loss == 0.0 {
                100.0
            } else {
                100.0 - (100.0 / (1.0 + gain / loss))
            }
        })
        .collect()
}

/// Calculate Bollinger Bands
pub struct BollingerBands {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub bandwidth: Vec<f64>,
    pub percent_b: Vec<f64>,
}

pub fn bollinger_bands(closes: &[f64], period: usize, std_dev: f64) -> BollingerBands {
    let middle = sma(closes, period);
    let std = rolling_std(closes, period);

    let mut upper = Vec::with_capacity(closes.len());
    let mut lower = Vec::with_capacity(closes.len());
    let mut bandwidth = Vec::with_capacity(closes.len());
    let mut percent_b = Vec::with_capacity(closes.len());

    for i in 0..closes.len() {
        if middle[i].is_nan() || std[i].is_nan() {
            upper.push(f64::NAN);
            lower.push(f64::NAN);
            bandwidth.push(f64::NAN);
            percent_b.push(f64::NAN);
        } else {
            let u = middle[i] + std_dev * std[i];
            let l = middle[i] - std_dev * std[i];
            upper.push(u);
            lower.push(l);

            let bw = if middle[i] != 0.0 {
                (u - l) / middle[i]
            } else {
                0.0
            };
            bandwidth.push(bw);

            let pb = if u != l {
                (closes[i] - l) / (u - l)
            } else {
                0.5
            };
            percent_b.push(pb);
        }
    }

    BollingerBands {
        upper,
        middle,
        lower,
        bandwidth,
        percent_b,
    }
}

/// Calculate Average True Range (ATR)
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

        let tr_value = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        tr.push(tr_value);
    }

    ema(&tr, period)
}

/// Calculate MACD (Moving Average Convergence Divergence)
pub struct MACD {
    pub macd_line: Vec<f64>,
    pub signal_line: Vec<f64>,
    pub histogram: Vec<f64>,
}

pub fn macd(closes: &[f64], fast: usize, slow: usize, signal: usize) -> MACD {
    let fast_ema = ema(closes, fast);
    let slow_ema = ema(closes, slow);

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

    let signal_line = ema(&macd_line, signal);

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

    MACD {
        macd_line,
        signal_line,
        histogram,
    }
}

/// Calculate price returns
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate log returns
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate rolling volatility (standard deviation of returns)
pub fn volatility(prices: &[f64], window: usize) -> Vec<f64> {
    let rets = returns(prices);
    let mut result = vec![f64::NAN];
    result.extend(rolling_std(&rets, window));
    result
}

/// Calculate volume ratio (current volume / rolling average volume)
pub fn volume_ratio(volumes: &[f64], window: usize) -> Vec<f64> {
    let avg_vol = rolling_mean(volumes, window);

    volumes
        .iter()
        .zip(avg_vol.iter())
        .map(|(&v, &avg)| {
            if avg.is_nan() || avg == 0.0 {
                f64::NAN
            } else {
                v / avg
            }
        })
        .collect()
}

/// Calculate price range ratio
pub fn range_ratio(highs: &[f64], lows: &[f64], closes: &[f64], window: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() {
        return vec![];
    }

    let ranges: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((&h, &l), &c)| if c > 0.0 { (h - l) / c } else { 0.0 })
        .collect();

    let avg_range = rolling_mean(&ranges, window);

    ranges
        .iter()
        .zip(avg_range.iter())
        .map(|(&r, &avg)| {
            if avg.is_nan() || avg == 0.0 {
                f64::NAN
            } else {
                r / avg
            }
        })
        .collect()
}

/// Calculate skewness of returns over a rolling window
pub fn rolling_skewness(data: &[f64], window: usize) -> Vec<f64> {
    if window < 3 || data.len() < window {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in window - 1..data.len() {
        let slice = &data[i + 1 - window..=i];
        let n = window as f64;

        let mean: f64 = slice.iter().sum::<f64>() / n;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            result.push(0.0);
        } else {
            let skew: f64 = slice.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
            result.push(skew);
        }
    }

    result
}

/// Calculate kurtosis of returns over a rolling window
pub fn rolling_kurtosis(data: &[f64], window: usize) -> Vec<f64> {
    if window < 4 || data.len() < window {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in window - 1..data.len() {
        let slice = &data[i + 1 - window..=i];
        let n = window as f64;

        let mean: f64 = slice.iter().sum::<f64>() / n;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            result.push(0.0);
        } else {
            let kurt: f64 = slice.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n - 3.0;
            result.push(kurt);
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

        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let result = ema(&data, 3);

        assert!(!result[2].is_nan());
        // EMA should be less reactive than the latest price
        assert!(result[5] < 15.0);
        assert!(result[5] > result[4]);
    }

    #[test]
    fn test_rsi() {
        // Steadily increasing prices should give RSI close to 100
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let result = rsi(&prices, 5);

        let last_rsi = result.last().unwrap();
        assert!(*last_rsi > 80.0);
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let result = returns(&prices);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.1).abs() < 1e-10);
        assert!((result[1] - (-0.0454545)).abs() < 1e-5);
    }
}
