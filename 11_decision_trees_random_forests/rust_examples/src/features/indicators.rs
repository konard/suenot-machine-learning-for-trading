//! Technical indicators implementation

use crate::data::Candle;

/// Simple Moving Average
pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
    if values.len() < period {
        return vec![f64::NAN; values.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..values.len() {
        let sum: f64 = values[(i + 1 - period)..=i].iter().sum();
        result.push(sum / period as f64);
    }

    result
}

/// Exponential Moving Average
pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; values.len()];

    // Start with SMA
    if values.len() >= period {
        let initial_sma: f64 = values[..period].iter().sum::<f64>() / period as f64;
        result[period - 1] = initial_sma;

        for i in period..values.len() {
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1];
        }
    }

    result
}

/// Relative Strength Index
pub fn rsi(candles: &[Candle], period: usize) -> Vec<f64> {
    if candles.len() < period + 1 {
        return vec![f64::NAN; candles.len()];
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..candles.len() {
        let change = candles[i].close - candles[i - 1].close;
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    let mut result = vec![f64::NAN; candles.len()];

    if gains.len() >= period {
        let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        for i in period..=gains.len() {
            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }

            if i < gains.len() {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }
        }
    }

    result
}

/// MACD (Moving Average Convergence Divergence)
pub struct MacdResult {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

pub fn macd(candles: &[Candle], fast: usize, slow: usize, signal: usize) -> MacdResult {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let ema_fast = ema(&closes, fast);
    let ema_slow = ema(&closes, slow);

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = ema(&macd_line, signal);

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    MacdResult {
        macd: macd_line,
        signal: signal_line,
        histogram,
    }
}

/// Bollinger Bands
pub struct BollingerBands {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub bandwidth: Vec<f64>,
    pub percent_b: Vec<f64>,
}

pub fn bollinger_bands(candles: &[Candle], period: usize, std_dev: f64) -> BollingerBands {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let middle = sma(&closes, period);

    let mut upper = vec![f64::NAN; closes.len()];
    let mut lower = vec![f64::NAN; closes.len()];
    let mut bandwidth = vec![f64::NAN; closes.len()];
    let mut percent_b = vec![f64::NAN; closes.len()];

    for i in (period - 1)..closes.len() {
        let slice = &closes[(i + 1 - period)..=i];
        let mean = middle[i];
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        upper[i] = mean + std_dev * std;
        lower[i] = mean - std_dev * std;

        if mean != 0.0 {
            bandwidth[i] = (upper[i] - lower[i]) / mean;
        }

        let band_width = upper[i] - lower[i];
        if band_width != 0.0 {
            percent_b[i] = (closes[i] - lower[i]) / band_width;
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

/// Average True Range
pub fn atr(candles: &[Candle], period: usize) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![f64::NAN; candles.len()];
    }

    let mut tr = vec![f64::NAN; candles.len()];

    // First TR
    tr[0] = candles[0].high - candles[0].low;

    // Calculate True Range
    for i in 1..candles.len() {
        let high_low = candles[i].high - candles[i].low;
        let high_prev_close = (candles[i].high - candles[i - 1].close).abs();
        let low_prev_close = (candles[i].low - candles[i - 1].close).abs();

        tr[i] = high_low.max(high_prev_close).max(low_prev_close);
    }

    // Calculate ATR using EMA
    ema(&tr, period)
}

/// Stochastic Oscillator
pub struct StochasticResult {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

pub fn stochastic(candles: &[Candle], k_period: usize, d_period: usize) -> StochasticResult {
    let mut k = vec![f64::NAN; candles.len()];

    for i in (k_period - 1)..candles.len() {
        let slice = &candles[(i + 1 - k_period)..=i];
        let highest: f64 = slice.iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let lowest: f64 = slice.iter().map(|c| c.low).fold(f64::MAX, f64::min);

        if highest != lowest {
            k[i] = ((candles[i].close - lowest) / (highest - lowest)) * 100.0;
        } else {
            k[i] = 50.0;
        }
    }

    let d = sma(&k, d_period);

    StochasticResult { k, d }
}

/// Volume Weighted Average Price (VWAP)
pub fn vwap(candles: &[Candle]) -> Vec<f64> {
    let mut result = vec![f64::NAN; candles.len()];
    let mut cumulative_tp_volume = 0.0;
    let mut cumulative_volume = 0.0;

    for (i, candle) in candles.iter().enumerate() {
        let typical_price = candle.typical_price();
        cumulative_tp_volume += typical_price * candle.volume;
        cumulative_volume += candle.volume;

        if cumulative_volume > 0.0 {
            result[i] = cumulative_tp_volume / cumulative_volume;
        }
    }

    result
}

/// On-Balance Volume
pub fn obv(candles: &[Candle]) -> Vec<f64> {
    let mut result = vec![0.0; candles.len()];

    if candles.is_empty() {
        return result;
    }

    result[0] = candles[0].volume;

    for i in 1..candles.len() {
        if candles[i].close > candles[i - 1].close {
            result[i] = result[i - 1] + candles[i].volume;
        } else if candles[i].close < candles[i - 1].close {
            result[i] = result[i - 1] - candles[i].volume;
        } else {
            result[i] = result[i - 1];
        }
    }

    result
}

/// Price momentum (rate of change)
pub fn momentum(values: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; values.len()];

    for i in period..values.len() {
        if values[i - period] != 0.0 {
            result[i] = (values[i] - values[i - period]) / values[i - period];
        }
    }

    result
}

/// Volatility (standard deviation of returns)
pub fn volatility(candles: &[Candle], period: usize) -> Vec<f64> {
    if candles.len() < period + 1 {
        return vec![f64::NAN; candles.len()];
    }

    let mut returns = vec![0.0; candles.len()];
    for i in 1..candles.len() {
        if candles[i - 1].close != 0.0 {
            returns[i] = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;
        }
    }

    let mut result = vec![f64::NAN; candles.len()];

    for i in period..candles.len() {
        let slice = &returns[(i + 1 - period)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / period as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        result[i] = variance.sqrt();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&values, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let candles: Vec<Candle> = (0..20)
            .map(|i| Candle::new(i, 100.0 + i as f64, 101.0, 99.0, 100.0 + i as f64, 1000.0))
            .collect();

        let result = rsi(&candles, 14);
        assert!(!result.last().unwrap().is_nan());
    }
}
