//! Simple Moving Average Crossover Strategy.

use crate::models::Candle;
use crate::strategies::{Signal, Strategy};
use crate::utils::sma;

/// SMA Crossover Strategy.
///
/// Generates buy signals when the fast SMA crosses above the slow SMA,
/// and sell signals when the fast SMA crosses below the slow SMA.
#[derive(Debug, Clone)]
pub struct SmaCrossover {
    /// Fast SMA period
    fast_period: usize,
    /// Slow SMA period
    slow_period: usize,
    /// Previous fast SMA value
    prev_fast: Option<f64>,
    /// Previous slow SMA value
    prev_slow: Option<f64>,
}

impl SmaCrossover {
    /// Create a new SMA Crossover strategy.
    ///
    /// # Arguments
    /// * `fast_period` - Period for the fast moving average (e.g., 10)
    /// * `slow_period` - Period for the slow moving average (e.g., 50)
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        assert!(fast_period < slow_period, "Fast period must be less than slow period");
        Self {
            fast_period,
            slow_period,
            prev_fast: None,
            prev_slow: None,
        }
    }

    /// Create with default parameters (10/50).
    pub fn default_params() -> Self {
        Self::new(10, 50)
    }
}

impl Strategy for SmaCrossover {
    fn name(&self) -> &str {
        "SMA Crossover"
    }

    fn on_candle(&mut self, _candle: &Candle, historical: &[Candle]) -> Signal {
        if historical.len() < self.slow_period {
            return Signal::Hold;
        }

        // Extract closing prices
        let closes: Vec<f64> = historical.iter().map(|c| c.close).collect();

        // Calculate SMAs
        let fast_sma = sma(&closes, self.fast_period);
        let slow_sma = sma(&closes, self.slow_period);

        let curr_fast = fast_sma.last().copied().flatten();
        let curr_slow = slow_sma.last().copied().flatten();

        let signal = match (curr_fast, curr_slow, self.prev_fast, self.prev_slow) {
            (Some(cf), Some(cs), Some(pf), Some(ps)) => {
                // Golden cross: fast crosses above slow
                if pf <= ps && cf > cs {
                    Signal::Buy(1.0)
                }
                // Death cross: fast crosses below slow
                else if pf >= ps && cf < cs {
                    Signal::Sell(1.0)
                }
                // Trend continuation
                else if cf > cs {
                    Signal::Hold // Already long or waiting
                } else {
                    Signal::Hold
                }
            }
            _ => Signal::Hold,
        };

        // Store current values for next iteration
        self.prev_fast = curr_fast;
        self.prev_slow = curr_slow;

        signal
    }

    fn reset(&mut self) {
        self.prev_fast = None;
        self.prev_slow = None;
    }

    fn min_history(&self) -> usize {
        self.slow_period
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
    fn test_sma_crossover_signals() {
        let mut strategy = SmaCrossover::new(3, 5);

        // Create uptrending data
        let candles: Vec<Candle> = (0..20)
            .map(|i| make_candle(100.0 + i as f64 * 2.0))
            .collect();

        // Run through candles
        let mut signals = Vec::new();
        for (i, candle) in candles.iter().enumerate() {
            let signal = strategy.on_candle(candle, &candles[..=i]);
            signals.push(signal);
        }

        // Should have a buy signal when fast crosses above slow
        assert!(signals.iter().any(|s| s.is_buy()));
    }
}
