//! Trading module for pattern recognition and signal generation
//!
//! This module provides tools for encoding market data into patterns
//! suitable for Hopfield Networks and generating trading signals
//! from pattern matches.

pub mod patterns;
pub mod signals;

use crate::data::Candle;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Strong uptrend
    BullTrend,
    /// Strong downtrend
    BearTrend,
    /// Sideways/ranging market
    Ranging,
    /// High volatility regime
    HighVolatility,
    /// Low volatility regime
    LowVolatility,
    /// Breakout from consolidation
    Breakout,
    /// Unknown/transitional
    Unknown,
}

impl MarketRegime {
    /// Get recommended position bias for this regime
    pub fn position_bias(&self) -> f64 {
        match self {
            MarketRegime::BullTrend => 1.0,
            MarketRegime::BearTrend => -1.0,
            MarketRegime::Ranging => 0.0,
            MarketRegime::HighVolatility => 0.0,
            MarketRegime::LowVolatility => 0.0,
            MarketRegime::Breakout => 0.5,
            MarketRegime::Unknown => 0.0,
        }
    }

    /// Get risk multiplier for this regime
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            MarketRegime::BullTrend => 1.0,
            MarketRegime::BearTrend => 1.0,
            MarketRegime::Ranging => 0.5,
            MarketRegime::HighVolatility => 0.3,
            MarketRegime::LowVolatility => 1.2,
            MarketRegime::Breakout => 0.7,
            MarketRegime::Unknown => 0.5,
        }
    }
}

/// Candlestick pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandlePattern {
    /// Strong bullish candle
    BullishEngulfing,
    /// Strong bearish candle
    BearishEngulfing,
    /// Doji (indecision)
    Doji,
    /// Hammer (potential reversal)
    Hammer,
    /// Shooting star (potential reversal)
    ShootingStar,
    /// Morning star (bullish reversal)
    MorningStar,
    /// Evening star (bearish reversal)
    EveningStar,
    /// Three white soldiers
    ThreeWhiteSoldiers,
    /// Three black crows
    ThreeBlackCrows,
    /// No specific pattern
    None,
}

/// Detect single candle patterns
pub fn detect_candle_pattern(candle: &Candle) -> CandlePattern {
    let body = candle.body_size();
    let range = candle.range();
    let upper_shadow = candle.upper_shadow();
    let lower_shadow = candle.lower_shadow();

    // Doji detection (small body relative to range)
    if range > 0.0 && body / range < 0.1 {
        return CandlePattern::Doji;
    }

    // Hammer detection (long lower shadow, small upper shadow)
    if lower_shadow > body * 2.0 && upper_shadow < body * 0.5 && candle.is_bullish() {
        return CandlePattern::Hammer;
    }

    // Shooting star detection (long upper shadow, small lower shadow)
    if upper_shadow > body * 2.0 && lower_shadow < body * 0.5 && candle.is_bearish() {
        return CandlePattern::ShootingStar;
    }

    CandlePattern::None
}

/// Detect multi-candle patterns
pub fn detect_multi_candle_pattern(candles: &[Candle]) -> CandlePattern {
    if candles.len() < 2 {
        return CandlePattern::None;
    }

    let len = candles.len();

    // Engulfing pattern detection
    if len >= 2 {
        let prev = &candles[len - 2];
        let curr = &candles[len - 1];

        // Bullish engulfing
        if prev.is_bearish()
            && curr.is_bullish()
            && curr.open < prev.close
            && curr.close > prev.open
        {
            return CandlePattern::BullishEngulfing;
        }

        // Bearish engulfing
        if prev.is_bullish()
            && curr.is_bearish()
            && curr.open > prev.close
            && curr.close < prev.open
        {
            return CandlePattern::BearishEngulfing;
        }
    }

    // Three white soldiers / Three black crows
    if len >= 3 {
        let c1 = &candles[len - 3];
        let c2 = &candles[len - 2];
        let c3 = &candles[len - 1];

        // Three white soldiers
        if c1.is_bullish()
            && c2.is_bullish()
            && c3.is_bullish()
            && c2.close > c1.close
            && c3.close > c2.close
        {
            return CandlePattern::ThreeWhiteSoldiers;
        }

        // Three black crows
        if c1.is_bearish()
            && c2.is_bearish()
            && c3.is_bearish()
            && c2.close < c1.close
            && c3.close < c2.close
        {
            return CandlePattern::ThreeBlackCrows;
        }
    }

    CandlePattern::None
}

/// Calculate simple moving average
pub fn sma(values: &[f64], period: usize) -> Option<f64> {
    if values.len() < period {
        return None;
    }

    let sum: f64 = values.iter().rev().take(period).sum();
    Some(sum / period as f64)
}

/// Calculate exponential moving average
pub fn ema(values: &[f64], period: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_value = values[0];

    for &value in values.iter().skip(1) {
        ema_value = (value - ema_value) * multiplier + ema_value;
    }

    Some(ema_value)
}

/// Calculate RSI (Relative Strength Index)
pub fn rsi(closes: &[f64], period: usize) -> Option<f64> {
    if closes.len() < period + 1 {
        return None;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..closes.len() {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(change.abs());
        }
    }

    let avg_gain: f64 = gains.iter().rev().take(period).sum::<f64>() / period as f64;
    let avg_loss: f64 = losses.iter().rev().take(period).sum::<f64>() / period as f64;

    if avg_loss < 1e-10 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - (100.0 / (1.0 + rs)))
}

/// Calculate standard deviation
pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    variance.sqrt()
}

/// Calculate volatility (standard deviation of returns)
pub fn volatility(closes: &[f64]) -> f64 {
    if closes.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    std_dev(&returns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle::new(Utc::now(), open, high, low, close, 1000.0, 100000.0)
    }

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sma(&values, 3), Some(4.0)); // (3+4+5)/3
        assert_eq!(sma(&values, 5), Some(3.0)); // (1+2+3+4+5)/5
    }

    #[test]
    fn test_rsi() {
        let closes = vec![44.0, 44.5, 44.25, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.25, 44.5, 45.0, 45.5, 46.0, 46.5];
        let rsi_value = rsi(&closes, 14).unwrap();
        assert!(rsi_value > 0.0 && rsi_value < 100.0);
    }

    #[test]
    fn test_detect_doji() {
        let doji = make_candle(100.0, 102.0, 98.0, 100.1);
        assert_eq!(detect_candle_pattern(&doji), CandlePattern::Doji);
    }

    #[test]
    fn test_detect_hammer() {
        let hammer = make_candle(100.0, 101.0, 95.0, 100.5);
        assert_eq!(detect_candle_pattern(&hammer), CandlePattern::Hammer);
    }

    #[test]
    fn test_volatility() {
        let closes = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        let vol = volatility(&closes);
        assert!(vol > 0.0);
    }
}
