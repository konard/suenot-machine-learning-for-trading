//! Technical Indicators

use crate::data::Candle;

/// Technical indicator calculations
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate RSI
    pub fn rsi(candles: &[Candle], period: usize) -> f64 {
        if candles.len() < period + 1 { return 50.0; }
        let changes: Vec<f64> = candles.windows(2).map(|w| w[1].close - w[0].close).collect();
        let recent = &changes[changes.len().saturating_sub(period)..];
        let gains: f64 = recent.iter().filter(|&&c| c > 0.0).sum();
        let losses: f64 = recent.iter().filter(|&&c| c < 0.0).map(|c| c.abs()).sum();
        if losses < 1e-10 { 100.0 } else { 100.0 - 100.0 / (1.0 + gains / losses) }
    }

    /// Calculate MACD
    pub fn macd(candles: &[Candle], fast: usize, slow: usize, signal: usize) -> (f64, f64, f64) {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let ema_fast = Self::ema(&prices, fast);
        let ema_slow = Self::ema(&prices, slow);
        let macd_line = ema_fast - ema_slow;
        let signal_line = macd_line * (2.0 / (signal as f64 + 1.0));
        let histogram = macd_line - signal_line;
        (macd_line, signal_line, histogram)
    }

    /// Calculate EMA
    pub fn ema(values: &[f64], period: usize) -> f64 {
        if values.is_empty() { return 0.0; }
        let alpha = 2.0 / (period as f64 + 1.0);
        values.iter().fold(values[0], |acc, &x| alpha * x + (1.0 - alpha) * acc)
    }

    /// Calculate SMA
    pub fn sma(values: &[f64], period: usize) -> f64 {
        if values.len() < period { return 0.0; }
        values.iter().rev().take(period).sum::<f64>() / period as f64
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(candles: &[Candle], period: usize, std_mult: f64) -> (f64, f64, f64) {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let sma = Self::sma(&prices, period);
        let std = Self::std_dev(&prices, period);
        (sma + std_mult * std, sma, sma - std_mult * std)
    }

    fn std_dev(values: &[f64], period: usize) -> f64 {
        if values.len() < period { return 0.0; }
        let recent: Vec<f64> = values.iter().rev().take(period).cloned().collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        (recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64).sqrt()
    }
}
