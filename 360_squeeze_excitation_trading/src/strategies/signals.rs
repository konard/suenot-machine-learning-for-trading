//! Trading Signals and Direction
//!
//! This module defines the core signal types used across trading strategies.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trading direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// No position
    Neutral,
}

impl Direction {
    /// Convert to numeric value
    pub fn to_f64(&self) -> f64 {
        match self {
            Direction::Long => 1.0,
            Direction::Short => -1.0,
            Direction::Neutral => 0.0,
        }
    }

    /// Create from numeric value
    pub fn from_f64(value: f64, threshold: f64) -> Self {
        if value > threshold {
            Direction::Long
        } else if value < -threshold {
            Direction::Short
        } else {
            Direction::Neutral
        }
    }

    /// Check if this is an active position
    pub fn is_active(&self) -> bool {
        *self != Direction::Neutral
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Direction of the signal
    pub direction: Direction,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Feature attention weights (for interpretability)
    pub feature_attention: Option<Array1<f64>>,
    /// Timestamp (milliseconds)
    pub timestamp: Option<u64>,
    /// Raw model output
    pub raw_signal: f64,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(direction: Direction, strength: f64) -> Self {
        Self {
            direction,
            strength,
            confidence: strength,
            feature_attention: None,
            timestamp: None,
            raw_signal: direction.to_f64() * strength,
        }
    }

    /// Create from raw model output
    pub fn from_raw(raw: f64, threshold: f64) -> Self {
        let direction = Direction::from_f64(raw, threshold);
        let strength = raw.abs().min(1.0);

        Self {
            direction,
            strength,
            confidence: strength,
            feature_attention: None,
            timestamp: None,
            raw_signal: raw,
        }
    }

    /// Add attention weights to signal
    pub fn with_attention(mut self, attention: Array1<f64>) -> Self {
        self.feature_attention = Some(attention);
        self
    }

    /// Add timestamp to signal
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Check if signal suggests going long
    pub fn is_long(&self) -> bool {
        self.direction == Direction::Long
    }

    /// Check if signal suggests going short
    pub fn is_short(&self) -> bool {
        self.direction == Direction::Short
    }

    /// Check if signal is neutral
    pub fn is_neutral(&self) -> bool {
        self.direction == Direction::Neutral
    }

    /// Get position size recommendation based on signal strength
    pub fn position_size(&self, max_size: f64) -> f64 {
        self.direction.to_f64() * self.strength * max_size
    }

    /// Get top attended features
    pub fn top_features(&self, feature_names: &[&str], k: usize) -> Vec<(String, f64)> {
        match &self.feature_attention {
            Some(attention) => {
                let mut indexed: Vec<(usize, f64)> = attention
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| (i, w))
                    .collect();

                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                indexed
                    .into_iter()
                    .take(k)
                    .map(|(i, w)| {
                        let name = feature_names.get(i).unwrap_or(&"unknown");
                        (name.to_string(), w)
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }
}

/// Position state for tracking
#[derive(Debug, Clone)]
pub struct Position {
    /// Current direction
    pub direction: Direction,
    /// Entry price
    pub entry_price: f64,
    /// Position size
    pub size: f64,
    /// Entry timestamp
    pub entry_time: u64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
}

impl Position {
    /// Create a new position
    pub fn new(direction: Direction, entry_price: f64, size: f64, entry_time: u64) -> Self {
        Self {
            direction,
            entry_price,
            size,
            entry_time,
            unrealized_pnl: 0.0,
        }
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_change = current_price - self.entry_price;
        self.unrealized_pnl = self.direction.to_f64() * price_change * self.size;
    }

    /// Calculate return percentage
    pub fn return_pct(&self, current_price: f64) -> f64 {
        let price_change_pct = (current_price - self.entry_price) / self.entry_price;
        self.direction.to_f64() * price_change_pct * 100.0
    }

    /// Check if stop loss is triggered
    pub fn check_stop_loss(&self, current_price: f64, stop_pct: f64) -> bool {
        self.return_pct(current_price) < -stop_pct
    }

    /// Check if take profit is triggered
    pub fn check_take_profit(&self, current_price: f64, take_profit_pct: f64) -> bool {
        self.return_pct(current_price) > take_profit_pct
    }
}

/// Signal filter to reduce noise
#[derive(Debug, Clone)]
pub struct SignalFilter {
    /// Minimum strength threshold
    min_strength: f64,
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Cooldown period (in bars)
    cooldown: usize,
    /// Bars since last signal
    bars_since_signal: usize,
}

impl SignalFilter {
    /// Create a new signal filter
    pub fn new(min_strength: f64, min_confidence: f64, cooldown: usize) -> Self {
        Self {
            min_strength,
            min_confidence,
            cooldown,
            bars_since_signal: cooldown, // Allow immediate first signal
        }
    }

    /// Filter a signal
    pub fn filter(&mut self, signal: &TradingSignal) -> Option<TradingSignal> {
        self.bars_since_signal += 1;

        // Check thresholds
        if signal.strength < self.min_strength || signal.confidence < self.min_confidence {
            return None;
        }

        // Check cooldown
        if self.bars_since_signal < self.cooldown {
            return None;
        }

        // Signal passes filter
        if signal.direction.is_active() {
            self.bars_since_signal = 0;
            Some(signal.clone())
        } else {
            None
        }
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.bars_since_signal = self.cooldown;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_conversion() {
        assert_eq!(Direction::Long.to_f64(), 1.0);
        assert_eq!(Direction::Short.to_f64(), -1.0);
        assert_eq!(Direction::Neutral.to_f64(), 0.0);

        assert_eq!(Direction::from_f64(0.5, 0.3), Direction::Long);
        assert_eq!(Direction::from_f64(-0.5, 0.3), Direction::Short);
        assert_eq!(Direction::from_f64(0.1, 0.3), Direction::Neutral);
    }

    #[test]
    fn test_signal_creation() {
        let signal = TradingSignal::from_raw(0.8, 0.2);
        assert!(signal.is_long());
        assert_eq!(signal.strength, 0.8);
    }

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new(Direction::Long, 100.0, 1.0, 0);
        pos.update_pnl(110.0);
        assert_eq!(pos.unrealized_pnl, 10.0);

        let mut short_pos = Position::new(Direction::Short, 100.0, 1.0, 0);
        short_pos.update_pnl(90.0);
        assert_eq!(short_pos.unrealized_pnl, 10.0);
    }

    #[test]
    fn test_signal_filter() {
        let mut filter = SignalFilter::new(0.3, 0.3, 5);

        // First signal should pass
        let signal = TradingSignal::from_raw(0.5, 0.2);
        assert!(filter.filter(&signal).is_some());

        // Next signal should be blocked by cooldown
        let signal2 = TradingSignal::from_raw(0.6, 0.2);
        assert!(filter.filter(&signal2).is_none());
    }
}
