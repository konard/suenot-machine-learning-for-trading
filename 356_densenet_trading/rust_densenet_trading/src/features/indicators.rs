//! Technical Indicators
//!
//! Common technical analysis indicators for feature engineering.

use crate::data::candle::CandleSeries;

/// Technical indicators calculator
#[derive(Debug, Clone)]
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate Simple Moving Average
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period || period == 0 {
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
        if prices.is_empty() || period == 0 {
            return vec![f64::NAN; prices.len()];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(prices.len());

        // First EMA is SMA
        let first_sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;
        result.extend(vec![f64::NAN; period - 1]);
        result.push(first_sma);

        // Calculate rest of EMA
        for i in period..prices.len() {
            let ema = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
            result.push(ema);
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 || period == 0 {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period];

        // Calculate price changes
        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        // Initial average gain/loss
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for &change in changes.iter().take(period) {
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss -= change;
            }
        }

        avg_gain /= period as f64;
        avg_loss /= period as f64;

        // First RSI
        let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 100.0 };
        result.push(100.0 - 100.0 / (1.0 + rs));

        // Smoothed RSI
        for i in period..changes.len() {
            let change = changes[i];
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };

            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 100.0 };
            result.push(100.0 - 100.0 / (1.0 + rs));
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        // MACD line
        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(&f, &s)| f - s)
            .collect();

        // Signal line (EMA of MACD)
        let signal_line = Self::ema(&macd_line, signal);

        // Histogram
        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(&m, &s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, period);

        let mut upper = Vec::with_capacity(prices.len());
        let mut lower = Vec::with_capacity(prices.len());

        for i in 0..prices.len() {
            if i < period - 1 {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                let window = &prices[(i + 1 - period)..=i];
                let mean = sma[i];
                let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
                let std = variance.sqrt();

                upper.push(mean + std_dev * std);
                lower.push(mean - std_dev * std);
            }
        }

        (upper, sma, lower)
    }

    /// Calculate ATR (Average True Range)
    pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        if highs.len() != lows.len() || lows.len() != closes.len() || highs.len() < 2 {
            return vec![f64::NAN; highs.len()];
        }

        // Calculate True Range
        let mut tr = vec![highs[0] - lows[0]];
        for i in 1..highs.len() {
            let h_l = highs[i] - lows[i];
            let h_cp = (highs[i] - closes[i - 1]).abs();
            let l_cp = (lows[i] - closes[i - 1]).abs();
            tr.push(h_l.max(h_cp).max(l_cp));
        }

        // Smooth with EMA
        Self::ema(&tr, period)
    }

    /// Calculate Stochastic Oscillator
    pub fn stochastic(highs: &[f64], lows: &[f64], closes: &[f64], k_period: usize, d_period: usize) -> (Vec<f64>, Vec<f64>) {
        if highs.len() != lows.len() || lows.len() != closes.len() || highs.len() < k_period {
            return (vec![f64::NAN; highs.len()], vec![f64::NAN; highs.len()]);
        }

        let mut k = vec![f64::NAN; k_period - 1];

        for i in (k_period - 1)..closes.len() {
            let window_high = highs[(i + 1 - k_period)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let window_low = lows[(i + 1 - k_period)..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            let range = window_high - window_low;
            if range > 0.0 {
                k.push((closes[i] - window_low) / range * 100.0);
            } else {
                k.push(50.0);
            }
        }

        let d = Self::sma(&k, d_period);

        (k, d)
    }

    /// Calculate OBV (On-Balance Volume)
    pub fn obv(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        if closes.len() != volumes.len() || closes.is_empty() {
            return vec![0.0; closes.len()];
        }

        let mut result = vec![volumes[0]];

        for i in 1..closes.len() {
            let prev_obv = result[i - 1];
            let obv = if closes[i] > closes[i - 1] {
                prev_obv + volumes[i]
            } else if closes[i] < closes[i - 1] {
                prev_obv - volumes[i]
            } else {
                prev_obv
            };
            result.push(obv);
        }

        result
    }

    /// Calculate ROC (Rate of Change)
    pub fn roc(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 || period == 0 {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..prices.len() {
            let change = (prices[i] - prices[i - period]) / prices[i - period] * 100.0;
            result.push(change);
        }

        result
    }

    /// Calculate CCI (Commodity Channel Index)
    pub fn cci(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        if highs.len() != lows.len() || lows.len() != closes.len() || highs.len() < period {
            return vec![f64::NAN; highs.len()];
        }

        // Typical Price
        let tp: Vec<f64> = highs
            .iter()
            .zip(lows.iter())
            .zip(closes.iter())
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();

        let sma_tp = Self::sma(&tp, period);

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..tp.len() {
            let window = &tp[(i + 1 - period)..=i];
            let mean = sma_tp[i];
            let mean_deviation: f64 = window.iter().map(|&x| (x - mean).abs()).sum::<f64>() / period as f64;

            if mean_deviation != 0.0 {
                result.push((tp[i] - mean) / (0.015 * mean_deviation));
            } else {
                result.push(0.0);
            }
        }

        result
    }

    /// Calculate Williams %R
    pub fn williams_r(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        if highs.len() != lows.len() || lows.len() != closes.len() || highs.len() < period {
            return vec![f64::NAN; highs.len()];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..closes.len() {
            let window_high = highs[(i + 1 - period)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let window_low = lows[(i + 1 - period)..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            let range = window_high - window_low;
            if range > 0.0 {
                result.push((window_high - closes[i]) / range * -100.0);
            } else {
                result.push(-50.0);
            }
        }

        result
    }

    /// Calculate log returns
    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN];
        for i in 1..prices.len() {
            result.push((prices[i] / prices[i - 1]).ln());
        }
        result
    }

    /// Calculate realized volatility
    pub fn realized_volatility(prices: &[f64], period: usize) -> Vec<f64> {
        let returns = Self::log_returns(prices);

        if returns.len() < period {
            return vec![f64::NAN; prices.len()];
        }

        let mut result = vec![f64::NAN; period];

        for i in period..returns.len() {
            let window = &returns[(i + 1 - period)..=i];
            let valid: Vec<f64> = window.iter().filter(|&&x| !x.is_nan()).cloned().collect();

            if valid.len() >= period / 2 {
                let mean: f64 = valid.iter().sum::<f64>() / valid.len() as f64;
                let variance: f64 = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
                result.push(variance.sqrt() * (252.0_f64).sqrt()); // Annualized
            } else {
                result.push(f64::NAN);
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
        assert!((sma[2] - 2.0).abs() < 0.001);
        assert!((sma[3] - 3.0).abs() < 0.001);
        assert!((sma[4] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_ema() {
        let prices = vec![22.0, 22.5, 23.0, 22.5, 23.5, 24.0, 23.5, 24.0];
        let ema = TechnicalIndicators::ema(&prices, 3);

        assert_eq!(ema.len(), prices.len());
        // First 2 values should be NaN
        assert!(ema[0].is_nan());
        assert!(ema[1].is_nan());
        // EMA should be calculated from index 2 onwards
        assert!(!ema[2].is_nan());
    }

    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5, 44.25, 44.0, 43.5, 44.0, 44.0, 44.0];
        let rsi = TechnicalIndicators::rsi(&prices, 14);

        assert_eq!(rsi.len(), prices.len());
        // RSI should be between 0 and 100
        for &val in rsi.iter().skip(14) {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }

    #[test]
    fn test_macd() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let (macd, signal, histogram) = TechnicalIndicators::macd(&prices, 12, 26, 9);

        assert_eq!(macd.len(), prices.len());
        assert_eq!(signal.len(), prices.len());
        assert_eq!(histogram.len(), prices.len());
    }

    #[test]
    fn test_bollinger_bands() {
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(&prices, 20, 2.0);

        for i in 19..prices.len() {
            assert!(upper[i] > middle[i]);
            assert!(middle[i] > lower[i]);
        }
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let returns = TechnicalIndicators::log_returns(&prices);

        assert!(returns[0].is_nan());
        assert!((returns[1] - (101.0_f64 / 100.0).ln()).abs() < 0.0001);
    }
}
