//! RSI (Relative Strength Index) Strategy.

use crate::models::Candle;
use crate::strategies::{Signal, Strategy};
use crate::utils::rsi;

/// RSI Mean Reversion Strategy.
///
/// Generates buy signals when RSI is oversold (below lower threshold),
/// and sell signals when RSI is overbought (above upper threshold).
#[derive(Debug, Clone)]
pub struct RsiStrategy {
    /// RSI period
    period: usize,
    /// Oversold threshold (buy when RSI below this)
    oversold: f64,
    /// Overbought threshold (sell when RSI above this)
    overbought: f64,
    /// Current position state
    in_position: bool,
}

impl RsiStrategy {
    /// Create a new RSI strategy.
    ///
    /// # Arguments
    /// * `period` - RSI calculation period (typically 14)
    /// * `oversold` - Oversold threshold (typically 30)
    /// * `overbought` - Overbought threshold (typically 70)
    pub fn new(period: usize, oversold: f64, overbought: f64) -> Self {
        assert!(oversold < overbought, "Oversold must be less than overbought");
        Self {
            period,
            oversold,
            overbought,
            in_position: false,
        }
    }

    /// Create with default parameters (14, 30, 70).
    pub fn default_params() -> Self {
        Self::new(14, 30.0, 70.0)
    }

    /// Create with aggressive parameters (14, 20, 80).
    pub fn aggressive() -> Self {
        Self::new(14, 20.0, 80.0)
    }
}

impl Strategy for RsiStrategy {
    fn name(&self) -> &str {
        "RSI Mean Reversion"
    }

    fn on_candle(&mut self, _candle: &Candle, historical: &[Candle]) -> Signal {
        if historical.len() < self.period + 1 {
            return Signal::Hold;
        }

        // Extract closing prices
        let closes: Vec<f64> = historical.iter().map(|c| c.close).collect();

        // Calculate RSI
        let rsi_values = rsi(&closes, self.period);

        let current_rsi = match rsi_values.last().copied().flatten() {
            Some(r) => r,
            None => return Signal::Hold,
        };

        // Generate signals based on RSI levels
        if !self.in_position && current_rsi < self.oversold {
            // Oversold - buy signal
            // Strength based on how oversold (more oversold = stronger signal)
            let strength = ((self.oversold - current_rsi) / self.oversold).min(1.0);
            self.in_position = true;
            Signal::Buy(0.5 + strength * 0.5) // 0.5 to 1.0 strength
        } else if self.in_position && current_rsi > self.overbought {
            // Overbought - sell signal
            self.in_position = false;
            Signal::Sell(1.0)
        } else if self.in_position && current_rsi > 50.0 {
            // Take partial profits when RSI returns to neutral
            Signal::Sell(0.5)
        } else {
            Signal::Hold
        }
    }

    fn reset(&mut self) {
        self.in_position = false;
    }

    fn min_history(&self) -> usize {
        self.period + 1
    }
}

/// RSI Trend Following Strategy.
///
/// Uses RSI to confirm trend direction and generates signals accordingly.
#[derive(Debug, Clone)]
pub struct RsiTrendStrategy {
    period: usize,
    /// RSI level above which we consider bullish
    bullish_level: f64,
    /// RSI level below which we consider bearish
    bearish_level: f64,
    prev_rsi: Option<f64>,
}

impl RsiTrendStrategy {
    pub fn new(period: usize, bullish_level: f64, bearish_level: f64) -> Self {
        Self {
            period,
            bullish_level,
            bearish_level,
            prev_rsi: None,
        }
    }

    pub fn default_params() -> Self {
        Self::new(14, 55.0, 45.0)
    }
}

impl Strategy for RsiTrendStrategy {
    fn name(&self) -> &str {
        "RSI Trend Following"
    }

    fn on_candle(&mut self, _candle: &Candle, historical: &[Candle]) -> Signal {
        if historical.len() < self.period + 1 {
            return Signal::Hold;
        }

        let closes: Vec<f64> = historical.iter().map(|c| c.close).collect();
        let rsi_values = rsi(&closes, self.period);

        let current_rsi = match rsi_values.last().copied().flatten() {
            Some(r) => r,
            None => return Signal::Hold,
        };

        let signal = match self.prev_rsi {
            Some(prev) => {
                // Buy when RSI crosses above bullish level
                if prev < self.bullish_level && current_rsi >= self.bullish_level {
                    Signal::Buy(1.0)
                }
                // Sell when RSI crosses below bearish level
                else if prev > self.bearish_level && current_rsi <= self.bearish_level {
                    Signal::Sell(1.0)
                } else {
                    Signal::Hold
                }
            }
            None => Signal::Hold,
        };

        self.prev_rsi = Some(current_rsi);
        signal
    }

    fn reset(&mut self) {
        self.prev_rsi = None;
    }

    fn min_history(&self) -> usize {
        self.period + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(close: f64) -> Candle {
        Candle::new(
            Utc::now(),
            "BTCUSDT".to_string(),
            close,
            close + 1.0,
            close - 1.0,
            close,
            1000.0,
        )
    }

    #[test]
    fn test_rsi_strategy_creation() {
        let strategy = RsiStrategy::default_params();
        assert_eq!(strategy.period, 14);
        assert_eq!(strategy.oversold, 30.0);
        assert_eq!(strategy.overbought, 70.0);
    }

    #[test]
    fn test_rsi_needs_warmup() {
        let mut strategy = RsiStrategy::default_params();

        // With not enough data, should return Hold
        let candles: Vec<Candle> = (0..10).map(|i| make_candle(100.0 + i as f64)).collect();
        let signal = strategy.on_candle(&candles[9], &candles);

        assert_eq!(signal, Signal::Hold);
    }
}
