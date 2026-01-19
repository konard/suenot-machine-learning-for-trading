//! Technical indicator calculations

use crate::api::Kline;

/// Technical features calculator
#[derive(Debug, Clone)]
pub struct Features;

impl Features {
    /// Calculates RSI (Relative Strength Index)
    pub fn rsi(closes: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.5; closes.len()]; // neutral RSI default

        if closes.len() < period + 1 {
            return result;
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

        // Initial averages
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..closes.len() {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - 100.0 / (1.0 + rs);
            } else if avg_gain > 0.0 {
                result[i] = 100.0;
            } else {
                result[i] = 50.0;
            }
        }

        // Normalize to [0, 1]
        result.iter().map(|&v| v / 100.0).collect()
    }

    /// Calculates ATR (Average True Range)
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; klines.len()];

        if klines.len() < 2 {
            return result;
        }

        // Calculate true ranges
        let mut trs = vec![0.0];
        for i in 1..klines.len() {
            let tr = klines[i].true_range(&klines[i - 1]);
            trs.push(tr);
        }

        // Calculate ATR using EMA
        if klines.len() >= period {
            let initial_atr: f64 = trs[1..=period].iter().sum::<f64>() / period as f64;
            result[period - 1] = initial_atr;

            let mut atr = initial_atr;
            for i in period..klines.len() {
                atr = (atr * (period - 1) as f64 + trs[i]) / period as f64;
                result[i] = atr;
            }
        }

        result
    }

    /// Calculates MACD (Moving Average Convergence Divergence)
    ///
    /// Returns (macd_line, signal_line, histogram)
    pub fn macd(
        closes: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = closes.len();
        let mut macd_line = vec![0.0; n];
        let mut signal_line = vec![0.0; n];
        let mut histogram = vec![0.0; n];

        if n < slow_period {
            return (macd_line, signal_line, histogram);
        }

        // Calculate EMAs
        let fast_ema = Self::ema(closes, fast_period);
        let slow_ema = Self::ema(closes, slow_period);

        // MACD line = fast EMA - slow EMA
        for i in 0..n {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }

        // Signal line = EMA of MACD line
        signal_line = Self::ema(&macd_line, signal_period);

        // Histogram = MACD line - signal line
        for i in 0..n {
            histogram[i] = macd_line[i] - signal_line[i];
        }

        (macd_line, signal_line, histogram)
    }

    /// Calculates EMA (Exponential Moving Average)
    pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; values.len()];

        if values.is_empty() || period == 0 {
            return result;
        }

        let multiplier = 2.0 / (period + 1) as f64;

        // First EMA is SMA
        if values.len() >= period {
            let sma: f64 = values[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = sma;

            for i in period..values.len() {
                result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }

        result
    }

    /// Calculates SMA (Simple Moving Average)
    pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; values.len()];

        if values.len() < period || period == 0 {
            return result;
        }

        let mut sum: f64 = values[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..values.len() {
            sum = sum - values[i - period] + values[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculates Bollinger Bands
    ///
    /// Returns (middle_band, upper_band, lower_band)
    pub fn bollinger_bands(
        closes: &[f64],
        period: usize,
        num_std: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = closes.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n < period {
            return (middle, upper, lower);
        }

        for i in (period - 1)..n {
            let window = &closes[(i + 1 - period)..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            middle[i] = mean;
            upper[i] = mean + num_std * std;
            lower[i] = mean - num_std * std;
        }

        (middle, upper, lower)
    }

    /// Calculates Bollinger Band %B
    pub fn bollinger_pct_b(closes: &[f64], period: usize, num_std: f64) -> Vec<f64> {
        let (_middle, upper, lower) = Self::bollinger_bands(closes, period, num_std);
        let mut result = vec![0.5; closes.len()]; // neutral default

        for i in 0..closes.len() {
            let range = upper[i] - lower[i];
            if range > 0.0 {
                result[i] = (closes[i] - lower[i]) / range;
            }
        }

        result
    }

    /// Calculates log returns
    pub fn log_returns(closes: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; closes.len()];

        for i in 1..closes.len() {
            if closes[i - 1] > 0.0 {
                result[i] = (closes[i] / closes[i - 1]).ln();
            }
        }

        result
    }

    /// Calculates rolling volatility (standard deviation of returns)
    pub fn rolling_volatility(closes: &[f64], period: usize) -> Vec<f64> {
        let returns = Self::log_returns(closes);
        let mut result = vec![0.0; closes.len()];

        if closes.len() < period {
            return result;
        }

        for i in (period - 1)..closes.len() {
            let window = &returns[(i + 1 - period)..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Calculates volume ratio (current volume / SMA of volume)
    pub fn volume_ratio(volumes: &[f64], period: usize) -> Vec<f64> {
        let sma = Self::sma(volumes, period);
        let mut result = vec![1.0; volumes.len()];

        for i in 0..volumes.len() {
            if sma[i] > 0.0 {
                result[i] = volumes[i] / sma[i];
            }
        }

        result
    }

    /// Normalizes values using z-score within a rolling window
    pub fn zscore(values: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; values.len()];

        if values.len() < period {
            return result;
        }

        for i in (period - 1)..values.len() {
            let window = &values[(i + 1 - period)..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            if std > 0.0 {
                result[i] = (values[i] - mean) / std;
            }
        }

        result
    }

    /// Clips values to a range
    pub fn clip(values: &[f64], min: f64, max: f64) -> Vec<f64> {
        values.iter().map(|&v| v.max(min).min(max)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sma = Features::sma(&values, 3);

        assert!((sma[2] - 2.0).abs() < 0.001); // (1+2+3)/3 = 2
        assert!((sma[9] - 9.0).abs() < 0.001); // (8+9+10)/3 = 9
    }

    #[test]
    fn test_ema() {
        let values = vec![10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0];
        let ema = Features::ema(&values, 3);

        // EMA should follow the trend
        assert!(ema[6] > ema[4]);
    }

    #[test]
    fn test_rsi() {
        // Uptrend should have high RSI
        let uptrend: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi_up = Features::rsi(&uptrend, 14);
        assert!(rsi_up.last().unwrap() > &0.7);

        // Downtrend should have low RSI
        let downtrend: Vec<f64> = (0..20).map(|i| 100.0 - i as f64).collect();
        let rsi_down = Features::rsi(&downtrend, 14);
        assert!(rsi_down.last().unwrap() < &0.3);
    }

    #[test]
    fn test_log_returns() {
        let closes = vec![100.0, 105.0, 103.0, 107.0];
        let returns = Features::log_returns(&closes);

        assert!((returns[1] - 0.04879).abs() < 0.001); // ln(105/100)
    }

    #[test]
    fn test_bollinger_bands() {
        let closes: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let (middle, upper, lower) = Features::bollinger_bands(&closes, 20, 2.0);

        // Upper should be above middle, lower should be below
        for i in 19..30 {
            assert!(upper[i] >= middle[i]);
            assert!(lower[i] <= middle[i]);
        }
    }

    #[test]
    fn test_zscore() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        let zs = Features::zscore(&values, 5);

        // Last value (100) should have positive z-score (above mean)
        // With window [6,7,8,9,100], mean=26, std≈37.8, z≈1.96
        assert!(zs[9] > 1.5);
        // First values should be 0 (before we have enough data)
        assert_eq!(zs[0], 0.0);
    }

    #[test]
    fn test_clip() {
        let values = vec![-5.0, 0.0, 5.0, 10.0, 15.0];
        let clipped = Features::clip(&values, 0.0, 10.0);

        assert_eq!(clipped, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }
}
