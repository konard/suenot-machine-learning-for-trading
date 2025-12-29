//! Technical indicators for feature engineering
//!
//! Implements common technical analysis indicators:
//! - Moving averages (SMA, EMA)
//! - RSI (Relative Strength Index)
//! - MACD (Moving Average Convergence Divergence)
//! - Bollinger Bands
//! - ATR (Average True Range)
//! - Volume-based indicators

use crate::data::Candle;

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate Simple Moving Average
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

    /// Calculate Exponential Moving Average
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let mut result = vec![f64::NAN; prices.len()];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // First EMA is SMA
        if prices.len() >= period {
            let first_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = first_sma;

            for i in period..prices.len() {
                result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; prices.len()];
        let mut gains = vec![0.0; prices.len()];
        let mut losses = vec![0.0; prices.len()];

        // Calculate gains and losses
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // First average
        let first_avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let first_avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        let mut avg_gain = first_avg_gain;
        let mut avg_loss = first_avg_loss;

        if avg_loss == 0.0 {
            result[period] = 100.0;
        } else {
            result[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        }

        // Subsequent RSI values using smoothed averages
        for i in (period + 1)..prices.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                result[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    /// Returns (macd_line, signal_line, histogram)
    pub fn macd(
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast_period);
        let ema_slow = Self::ema(prices, slow_period);

        // MACD line = Fast EMA - Slow EMA
        let mut macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        // Signal line = EMA of MACD line
        let valid_macd: Vec<f64> = macd_line.iter().filter(|x| !x.is_nan()).cloned().collect();
        let signal_ema = Self::ema(&valid_macd, signal_period);

        let mut signal_line = vec![f64::NAN; prices.len()];
        let start_idx = prices.len() - valid_macd.len();

        for (i, &val) in signal_ema.iter().enumerate() {
            if start_idx + i < signal_line.len() {
                signal_line[start_idx + i] = val;
            }
        }

        // Histogram = MACD line - Signal line
        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| {
                if m.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    m - s
                }
            })
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    /// Returns (middle_band, upper_band, lower_band)
    pub fn bollinger_bands(
        prices: &[f64],
        period: usize,
        num_std: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = Self::sma(prices, period);
        let mut upper = vec![f64::NAN; prices.len()];
        let mut lower = vec![f64::NAN; prices.len()];

        for i in (period - 1)..prices.len() {
            let slice = &prices[(i + 1 - period)..=i];
            let mean = middle[i];
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + num_std * std;
            lower[i] = mean - num_std * std;
        }

        (middle, upper, lower)
    }

    /// Calculate ATR (Average True Range)
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

        // ATR is SMA of TR
        Self::sma(&tr, period)
    }

    /// Calculate returns (percentage change)
    pub fn returns(prices: &[f64]) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let mut result = vec![f64::NAN];

        for i in 1..prices.len() {
            if prices[i - 1] != 0.0 {
                result.push((prices[i] - prices[i - 1]) / prices[i - 1]);
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

    /// Calculate rolling standard deviation
    pub fn rolling_std(values: &[f64], period: usize) -> Vec<f64> {
        if values.len() < period {
            return vec![f64::NAN; values.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..values.len() {
            let slice = &values[(i + 1 - period)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result.push(variance.sqrt());
        }

        result
    }

    /// Calculate rolling minimum
    pub fn rolling_min(values: &[f64], period: usize) -> Vec<f64> {
        if values.len() < period {
            return vec![f64::NAN; values.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..values.len() {
            let min = values[(i + 1 - period)..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            result.push(min);
        }

        result
    }

    /// Calculate rolling maximum
    pub fn rolling_max(values: &[f64], period: usize) -> Vec<f64> {
        if values.len() < period {
            return vec![f64::NAN; values.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..values.len() {
            let max = values[(i + 1 - period)..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            result.push(max);
        }

        result
    }

    /// Calculate OBV (On-Balance Volume)
    pub fn obv(candles: &[Candle]) -> Vec<f64> {
        if candles.is_empty() {
            return vec![];
        }

        let mut result = vec![0.0];

        for i in 1..candles.len() {
            let prev_obv = result[i - 1];
            let obv = if candles[i].close > candles[i - 1].close {
                prev_obv + candles[i].volume
            } else if candles[i].close < candles[i - 1].close {
                prev_obv - candles[i].volume
            } else {
                prev_obv
            };
            result.push(obv);
        }

        result
    }

    /// Calculate VWAP (Volume Weighted Average Price)
    pub fn vwap(candles: &[Candle]) -> Vec<f64> {
        let mut result = Vec::with_capacity(candles.len());
        let mut cumulative_tp_vol = 0.0;
        let mut cumulative_vol = 0.0;

        for candle in candles {
            let typical_price = (candle.high + candle.low + candle.close) / 3.0;
            cumulative_tp_vol += typical_price * candle.volume;
            cumulative_vol += candle.volume;

            if cumulative_vol > 0.0 {
                result.push(cumulative_tp_vol / cumulative_vol);
            } else {
                result.push(typical_price);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::sma(&prices, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let rsi = TechnicalIndicators::rsi(&prices, 14);

        // RSI should be between 0 and 100
        for &val in &rsi {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 99.0, 105.0];
        let returns = TechnicalIndicators::returns(&prices);

        assert!(returns[0].is_nan());
        assert!((returns[1] - 0.1).abs() < 1e-10);
        assert!((returns[2] - (-0.1)).abs() < 1e-10);
    }
}
