//! Технические индикаторы

use crate::types::Candle;

/// Simple Moving Average (SMA)
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..data.len() {
        let sum: f64 = data[i + 1 - period..=i].iter().sum();
        result.push(sum / period as f64);
    }

    result
}

/// Exponential Moving Average (EMA)
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Первое значение - SMA
    if data.len() >= period {
        let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result.extend(vec![f64::NAN; period - 1]);
        result.push(initial_sma);

        let mut prev_ema = initial_sma;
        for &value in &data[period..] {
            let current_ema = (value - prev_ema) * multiplier + prev_ema;
            result.push(current_ema);
            prev_ema = current_ema;
        }
    } else {
        result = vec![f64::NAN; data.len()];
    }

    result
}

/// Relative Strength Index (RSI)
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![f64::NAN; data.len()];
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        gains.push(change.max(0.0));
        losses.push((-change).max(0.0));
    }

    let mut result = vec![f64::NAN; period];

    // Первое среднее
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    for i in period..gains.len() {
        if avg_loss == 0.0 {
            result.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            result.push(100.0 - 100.0 / (1.0 + rs));
        }

        // Smoothed averages
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
    }

    // Добавляем NaN в начало чтобы длина совпадала
    let mut final_result = vec![f64::NAN];
    final_result.extend(result);
    final_result
}

/// MACD (Moving Average Convergence Divergence)
pub struct MACDResult {
    pub macd_line: Vec<f64>,
    pub signal_line: Vec<f64>,
    pub histogram: Vec<f64>,
}

pub fn macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> MACDResult {
    let ema_fast = ema(data, fast);
    let ema_slow = ema(data, slow);

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

    MACDResult {
        macd_line,
        signal_line,
        histogram,
    }
}

/// Bollinger Bands
pub struct BollingerBands {
    pub middle: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub bandwidth: Vec<f64>,
}

pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> BollingerBands {
    let middle = sma(data, period);
    let mut upper = Vec::with_capacity(data.len());
    let mut lower = Vec::with_capacity(data.len());
    let mut bandwidth = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        if i < period - 1 {
            upper.push(f64::NAN);
            lower.push(f64::NAN);
            bandwidth.push(f64::NAN);
            continue;
        }

        let window = &data[i + 1 - period..=i];
        let mean = window.iter().sum::<f64>() / period as f64;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        let mid = middle[i];
        upper.push(mid + std_dev * std);
        lower.push(mid - std_dev * std);
        bandwidth.push(4.0 * std_dev * std / mid);
    }

    BollingerBands {
        middle,
        upper,
        lower,
        bandwidth,
    }
}

/// Average True Range (ATR)
pub fn atr(candles: &[Candle], period: usize) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![f64::NAN; candles.len()];
    }

    // Вычисляем True Range
    let mut tr = vec![candles[0].high - candles[0].low];
    for i in 1..candles.len() {
        let current = &candles[i];
        let prev_close = candles[i - 1].close;
        tr.push(current.true_range(Some(prev_close)));
    }

    // ATR - это EMA от TR
    ema(&tr, period)
}

/// Stochastic Oscillator
pub struct StochasticResult {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

pub fn stochastic(candles: &[Candle], k_period: usize, d_period: usize) -> StochasticResult {
    let mut k = vec![f64::NAN; k_period - 1];

    for i in (k_period - 1)..candles.len() {
        let window = &candles[i + 1 - k_period..=i];
        let lowest_low = window.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let highest_high = window
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);

        let current_close = candles[i].close;
        let range = highest_high - lowest_low;

        if range > 0.0 {
            k.push(100.0 * (current_close - lowest_low) / range);
        } else {
            k.push(50.0);
        }
    }

    let d = sma(&k, d_period);

    StochasticResult { k, d }
}

/// On-Balance Volume (OBV)
pub fn obv(candles: &[Candle]) -> Vec<f64> {
    if candles.is_empty() {
        return Vec::new();
    }

    let mut result = vec![candles[0].volume];

    for i in 1..candles.len() {
        let prev_obv = result[i - 1];
        let current = &candles[i];
        let prev = &candles[i - 1];

        let new_obv = if current.close > prev.close {
            prev_obv + current.volume
        } else if current.close < prev.close {
            prev_obv - current.volume
        } else {
            prev_obv
        };

        result.push(new_obv);
    }

    result
}

/// Momentum
pub fn momentum(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        result.push(data[i] - data[i - period]);
    }

    result
}

/// Rate of Change (ROC)
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        if data[i - period] != 0.0 {
            result.push((data[i] - data[i - period]) / data[i - period] * 100.0);
        } else {
            result.push(f64::NAN);
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

        assert_eq!(result.len(), 5);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema(&data, 3);

        assert_eq!(result.len(), 10);
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_rsi() {
        let data: Vec<f64> = (0..20).map(|x| (x as f64).sin() * 10.0 + 50.0).collect();
        let result = rsi(&data, 14);

        assert_eq!(result.len(), 20);
        for val in result.iter().skip(14) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_bollinger_bands() {
        let data: Vec<f64> = (0..50).map(|x| 100.0 + (x as f64 * 0.1).sin() * 5.0).collect();
        let bb = bollinger_bands(&data, 20, 2.0);

        assert_eq!(bb.middle.len(), 50);
        for i in 20..50 {
            assert!(bb.upper[i] > bb.middle[i]);
            assert!(bb.lower[i] < bb.middle[i]);
        }
    }
}
