//! Technical indicators for feature engineering

use crate::utils::math::{rolling_mean, rolling_std};

/// Technical indicators calculator
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![];
        }

        let mut gains = Vec::with_capacity(prices.len() - 1);
        let mut losses = Vec::with_capacity(prices.len() - 1);

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

        let mut result = Vec::with_capacity(prices.len() - period);
        let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        for i in period..gains.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            let rs = if avg_loss > 0.0 {
                avg_gain / avg_loss
            } else {
                100.0
            };

            result.push(100.0 - 100.0 / (1.0 + rs));
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        let min_len = ema_fast.len().min(ema_slow.len());
        let offset = ema_fast.len() - min_len;

        let macd_line: Vec<f64> = ema_fast[offset..]
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = Self::ema(&macd_line, signal);
        let offset2 = macd_line.len() - signal_line.len();

        let histogram: Vec<f64> = macd_line[offset2..]
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd_line[offset2..].to_vec(), signal_line, histogram)
    }

    /// Calculate EMA (Exponential Moving Average)
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![];
        }

        let multiplier = 2.0 / (period + 1) as f64;
        let mut result = Vec::with_capacity(prices.len() - period + 1);

        // Start with SMA
        let sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
        result.push(sma);

        let mut prev_ema = sma;
        for price in &prices[period..] {
            let ema = (price - prev_ema) * multiplier + prev_ema;
            result.push(ema);
            prev_ema = ema;
        }

        result
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = rolling_mean(prices, period);
        let std = rolling_std(prices, period);

        let upper: Vec<f64> = middle.iter().zip(std.iter()).map(|(m, s)| m + num_std * s).collect();
        let lower: Vec<f64> = middle.iter().zip(std.iter()).map(|(m, s)| m - num_std * s).collect();

        (upper, middle, lower)
    }

    /// Calculate ATR (Average True Range)
    pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        if high.len() < 2 || high.len() != low.len() || high.len() != close.len() {
            return vec![];
        }

        let mut true_ranges = Vec::with_capacity(high.len() - 1);

        for i in 1..high.len() {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            true_ranges.push(hl.max(hc).max(lc));
        }

        Self::ema(&true_ranges, period)
    }

    /// Calculate price momentum
    pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![];
        }

        prices
            .windows(period + 1)
            .map(|w| (w[period] - w[0]) / w[0])
            .collect()
    }

    /// Calculate VWAP deviation
    pub fn vwap_deviation(prices: &[f64], volumes: &[f64]) -> Vec<f64> {
        if prices.len() != volumes.len() || prices.is_empty() {
            return vec![];
        }

        let mut cum_pv = 0.0;
        let mut cum_v = 0.0;

        prices
            .iter()
            .zip(volumes.iter())
            .map(|(&p, &v)| {
                cum_pv += p * v;
                cum_v += v;
                let vwap = if cum_v > 0.0 { cum_pv / cum_v } else { p };
                (p - vwap) / vwap
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let rsi = TechnicalIndicators::rsi(&prices, 14);
        assert!(!rsi.is_empty());
        // Uptrend should have high RSI
        assert!(rsi.last().unwrap() > &50.0);
    }

    #[test]
    fn test_ema() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ema = TechnicalIndicators::ema(&prices, 5);
        assert!(!ema.is_empty());
    }

    #[test]
    fn test_bollinger() {
        let prices: Vec<f64> = (1..=30).map(|x| x as f64 + (x as f64 * 0.1).sin()).collect();
        let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(&prices, 20, 2.0);

        assert!(!upper.is_empty());
        // Upper should be above middle, middle above lower
        for i in 0..upper.len() {
            assert!(upper[i] >= middle[i]);
            assert!(middle[i] >= lower[i]);
        }
    }
}
