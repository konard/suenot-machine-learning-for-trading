//! Technical indicators for feature engineering
//!
//! This module provides various technical analysis indicators
//! commonly used in trading strategies.

use crate::data::Candle;

/// Simple Moving Average
pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![f64::NAN; prices.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..prices.len() {
        let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
        result.push(sum / period as f64);
    }

    result
}

/// Exponential Moving Average
pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; prices.len()];

    // Start with SMA for the first value
    if prices.len() >= period {
        let initial_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
        result[period - 1] = initial_sma;

        for i in period..prices.len() {
            result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
        }
    }

    result
}

/// Relative Strength Index (RSI)
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![f64::NAN; prices.len()];
    }

    let mut gains = Vec::with_capacity(prices.len());
    let mut losses = Vec::with_capacity(prices.len());

    gains.push(0.0);
    losses.push(0.0);

    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    let mut result = vec![f64::NAN; prices.len()];

    // Initial averages
    let initial_avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let initial_avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    let mut avg_gain = initial_avg_gain;
    let mut avg_loss = initial_avg_loss;

    for i in period..prices.len() {
        if i == period {
            // First RSI value
        } else {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        }

        if avg_loss == 0.0 {
            result[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    result
}

/// Moving Average Convergence Divergence (MACD)
pub struct MacdResult {
    pub macd_line: Vec<f64>,
    pub signal_line: Vec<f64>,
    pub histogram: Vec<f64>,
}

pub fn macd(prices: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> MacdResult {
    let fast_ema = ema(prices, fast_period);
    let slow_ema = ema(prices, slow_period);

    let macd_line: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();

    // Calculate signal line (EMA of MACD line)
    let valid_macd: Vec<f64> = macd_line.iter().filter(|x| !x.is_nan()).copied().collect();
    let signal_ema = ema(&valid_macd, signal_period);

    let mut signal_line = vec![f64::NAN; macd_line.len()];
    let mut signal_idx = 0;
    for i in 0..macd_line.len() {
        if !macd_line[i].is_nan() {
            if signal_idx < signal_ema.len() {
                signal_line[i] = signal_ema[signal_idx];
            }
            signal_idx += 1;
        }
    }

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    MacdResult {
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

pub fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> BollingerBands {
    let middle = sma(prices, period);
    let mut upper = vec![f64::NAN; prices.len()];
    let mut lower = vec![f64::NAN; prices.len()];
    let mut bandwidth = vec![f64::NAN; prices.len()];

    for i in (period - 1)..prices.len() {
        let slice = &prices[(i + 1 - period)..=i];
        let mean = middle[i];

        // Calculate standard deviation
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();

        upper[i] = mean + num_std * std_dev;
        lower[i] = mean - num_std * std_dev;
        bandwidth[i] = (upper[i] - lower[i]) / mean * 100.0;
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

    // Calculate True Range
    let mut tr = vec![candles[0].high - candles[0].low];
    for i in 1..candles.len() {
        let high_low = candles[i].high - candles[i].low;
        let high_close = (candles[i].high - candles[i - 1].close).abs();
        let low_close = (candles[i].low - candles[i - 1].close).abs();
        tr.push(high_low.max(high_close).max(low_close));
    }

    // Calculate ATR using EMA of TR
    ema(&tr, period)
}

/// On-Balance Volume (OBV)
pub fn obv(candles: &[Candle]) -> Vec<f64> {
    if candles.is_empty() {
        return vec![];
    }

    let mut result = vec![candles[0].volume];

    for i in 1..candles.len() {
        let prev_obv = result[i - 1];
        let new_obv = if candles[i].close > candles[i - 1].close {
            prev_obv + candles[i].volume
        } else if candles[i].close < candles[i - 1].close {
            prev_obv - candles[i].volume
        } else {
            prev_obv
        };
        result.push(new_obv);
    }

    result
}

/// Price Rate of Change (ROC)
pub fn roc(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; prices.len()];

    for i in period..prices.len() {
        if prices[i - period] != 0.0 {
            result[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100.0;
        }
    }

    result
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
        let highest_high = slice.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = slice.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);

        if highest_high != lowest_low {
            k[i] = (candles[i].close - lowest_low) / (highest_high - lowest_low) * 100.0;
        } else {
            k[i] = 50.0;
        }
    }

    let d = sma(&k, d_period);

    StochasticResult { k, d }
}

/// Commodity Channel Index (CCI)
pub fn cci(candles: &[Candle], period: usize) -> Vec<f64> {
    let typical_prices: Vec<f64> = candles
        .iter()
        .map(|c| (c.high + c.low + c.close) / 3.0)
        .collect();

    let sma_tp = sma(&typical_prices, period);
    let mut result = vec![f64::NAN; candles.len()];

    for i in (period - 1)..candles.len() {
        let slice = &typical_prices[(i + 1 - period)..=i];
        let mean = sma_tp[i];

        // Mean deviation
        let mean_deviation: f64 = slice.iter().map(|x| (x - mean).abs()).sum::<f64>() / period as f64;

        if mean_deviation != 0.0 {
            result[i] = (typical_prices[i] - mean) / (0.015 * mean_deviation);
        }
    }

    result
}

/// Williams %R
pub fn williams_r(candles: &[Candle], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; candles.len()];

    for i in (period - 1)..candles.len() {
        let slice = &candles[(i + 1 - period)..=i];
        let highest_high = slice.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = slice.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);

        if highest_high != lowest_low {
            result[i] = (highest_high - candles[i].close) / (highest_high - lowest_low) * -100.0;
        } else {
            result[i] = -50.0;
        }
    }

    result
}

/// Money Flow Index (MFI)
pub fn mfi(candles: &[Candle], period: usize) -> Vec<f64> {
    if candles.len() < period + 1 {
        return vec![f64::NAN; candles.len()];
    }

    // Calculate typical price and money flow
    let typical_prices: Vec<f64> = candles
        .iter()
        .map(|c| (c.high + c.low + c.close) / 3.0)
        .collect();

    let money_flow: Vec<f64> = candles
        .iter()
        .zip(typical_prices.iter())
        .map(|(c, tp)| tp * c.volume)
        .collect();

    let mut result = vec![f64::NAN; candles.len()];

    for i in period..candles.len() {
        let mut positive_flow = 0.0;
        let mut negative_flow = 0.0;

        for j in (i - period + 1)..=i {
            if typical_prices[j] > typical_prices[j - 1] {
                positive_flow += money_flow[j];
            } else if typical_prices[j] < typical_prices[j - 1] {
                negative_flow += money_flow[j];
            }
        }

        if negative_flow != 0.0 {
            let money_ratio = positive_flow / negative_flow;
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio));
        } else {
            result[i] = 100.0;
        }
    }

    result
}

/// Calculate returns (percentage change)
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let mut result = vec![f64::NAN];

    for i in 1..prices.len() {
        if prices[i - 1] != 0.0 {
            result.push((prices[i] - prices[i - 1]) / prices[i - 1] * 100.0);
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Calculate log returns
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let mut result = vec![f64::NAN];

    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 && prices[i] > 0.0 {
            result.push((prices[i] / prices[i - 1]).ln());
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Calculate rolling volatility (standard deviation of returns)
pub fn volatility(prices: &[f64], period: usize) -> Vec<f64> {
    let rets = returns(prices);
    let mut result = vec![f64::NAN; prices.len()];

    for i in period..prices.len() {
        let slice = &rets[(i + 1 - period)..=i];
        let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

        if valid.len() >= 2 {
            let mean: f64 = valid.iter().sum::<f64>() / valid.len() as f64;
            let variance: f64 = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
            result[i] = variance.sqrt();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&prices, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 43.75, 44.0, 44.5, 44.25, 44.75, 45.0, 45.5];
        let result = rsi(&prices, 14);

        // RSI should be between 0 and 100 for valid values
        for val in result.iter().skip(14) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let result = returns(&prices);

        assert!(result[0].is_nan());
        assert!((result[1] - 10.0).abs() < 1e-10);
        assert!((result[2] - (-4.545454545)).abs() < 1e-6);
    }
}
