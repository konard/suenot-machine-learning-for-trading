//! Technical indicators for crypto trading

use crate::data::types::Candle;

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate Simple Moving Average
    pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let sum: f64 = values[(i + 1 - period)..=i].iter().sum();
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate Exponential Moving Average
    pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = vec![values[0]];

        for i in 1..values.len() {
            let ema = alpha * values[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema);
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(candles: &[Candle], period: usize) -> Vec<f64> {
        if candles.len() < period + 1 {
            return vec![f64::NAN; candles.len()];
        }

        let n = candles.len();
        let mut result = vec![f64::NAN; n];

        // Calculate price changes
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = candles[i].close - candles[i - 1].close;
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate first average
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            result[period] = 100.0;
        }

        // Calculate subsequent values using smoothed average
        for i in (period + 1)..n {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                result[i] = 100.0;
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(
        candles: &[Candle],
        fast: usize,
        slow: usize,
        signal: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

        let ema_fast = Self::ema(&closes, fast);
        let ema_slow = Self::ema(&closes, slow);

        let n = closes.len();
        let mut macd_line = vec![f64::NAN; n];

        for i in 0..n {
            if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
                macd_line[i] = ema_fast[i] - ema_slow[i];
            }
        }

        // Signal line is EMA of MACD
        let signal_line = Self::ema(&macd_line, signal);

        // Histogram
        let mut histogram = vec![f64::NAN; n];
        for i in 0..n {
            if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                histogram[i] = macd_line[i] - signal_line[i];
            }
        }

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(
        candles: &[Candle],
        period: usize,
        std_dev: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let n = closes.len();

        let sma = Self::sma(&closes, period);
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let slice = &closes[(i + 1 - period)..=i];
            let mean = sma[i];
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }

        (upper, sma, lower)
    }

    /// Calculate ATR (Average True Range)
    pub fn atr(candles: &[Candle], period: usize) -> Vec<f64> {
        if candles.is_empty() {
            return vec![];
        }

        let n = candles.len();
        let mut true_ranges = vec![candles[0].high - candles[0].low];

        for i in 1..n {
            true_ranges.push(candles[i].true_range(Some(candles[i - 1].close)));
        }

        Self::ema(&true_ranges, period)
    }

    /// Calculate Stochastic Oscillator
    pub fn stochastic(candles: &[Candle], k_period: usize, d_period: usize) -> (Vec<f64>, Vec<f64>) {
        let n = candles.len();
        let mut k = vec![f64::NAN; n];

        for i in (k_period - 1)..n {
            let slice = &candles[(i + 1 - k_period)..=i];
            let high = slice.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
            let low = slice.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
            let close = candles[i].close;

            if high > low {
                k[i] = (close - low) / (high - low) * 100.0;
            }
        }

        let d = Self::sma(&k, d_period);

        (k, d)
    }

    /// Calculate momentum
    pub fn momentum(candles: &[Candle], period: usize) -> Vec<f64> {
        let n = candles.len();
        let mut result = vec![f64::NAN; n];

        for i in period..n {
            result[i] = candles[i].close - candles[i - period].close;
        }

        result
    }

    /// Calculate Rate of Change (ROC)
    pub fn roc(candles: &[Candle], period: usize) -> Vec<f64> {
        let n = candles.len();
        let mut result = vec![f64::NAN; n];

        for i in period..n {
            if candles[i - period].close != 0.0 {
                result[i] = (candles[i].close - candles[i - period].close)
                    / candles[i - period].close
                    * 100.0;
            }
        }

        result
    }

    /// Calculate volatility (standard deviation of returns)
    pub fn volatility(candles: &[Candle], period: usize) -> Vec<f64> {
        let returns: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();
        let n = returns.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let slice = &returns[(i + 1 - period)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (period - 1) as f64;
            result[i] = variance.sqrt();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles() -> Vec<Candle> {
        (0..50)
            .map(|i| Candle {
                timestamp: i * 1000,
                open: 100.0 + (i as f64 * 0.1),
                high: 101.0 + (i as f64 * 0.1),
                low: 99.0 + (i as f64 * 0.1),
                close: 100.5 + (i as f64 * 0.1),
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::sma(&values, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let candles = make_candles();
        let rsi = TechnicalIndicators::rsi(&candles, 14);

        // Should have NaN for first 14 values
        assert!(rsi[0].is_nan());
        assert!(!rsi[14].is_nan());
    }

    #[test]
    fn test_bollinger() {
        let candles = make_candles();
        let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(&candles, 20, 2.0);

        // Check that bands are ordered correctly
        for i in 19..candles.len() {
            assert!(upper[i] > middle[i]);
            assert!(middle[i] > lower[i]);
        }
    }
}
