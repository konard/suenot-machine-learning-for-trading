//! Spike Decoding Module
//!
//! Converts output spikes from SNN into trading signals.

pub mod trading;

use crate::neuron::SpikeEvent;

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Buy signal with confidence and urgency
    Buy { confidence: f64, urgency: f64 },
    /// Sell signal with confidence and urgency
    Sell { confidence: f64, urgency: f64 },
    /// Hold position
    Hold,
}

impl TradingSignal {
    /// Check if this is a buy signal
    pub fn is_buy(&self) -> bool {
        matches!(self, TradingSignal::Buy { .. })
    }

    /// Check if this is a sell signal
    pub fn is_sell(&self) -> bool {
        matches!(self, TradingSignal::Sell { .. })
    }

    /// Check if this is a hold signal
    pub fn is_hold(&self) -> bool {
        matches!(self, TradingSignal::Hold)
    }

    /// Get confidence if applicable
    pub fn confidence(&self) -> Option<f64> {
        match self {
            TradingSignal::Buy { confidence, .. } => Some(*confidence),
            TradingSignal::Sell { confidence, .. } => Some(*confidence),
            TradingSignal::Hold => None,
        }
    }

    /// Get urgency if applicable
    pub fn urgency(&self) -> Option<f64> {
        match self {
            TradingSignal::Buy { urgency, .. } => Some(*urgency),
            TradingSignal::Sell { urgency, .. } => Some(*urgency),
            TradingSignal::Hold => None,
        }
    }
}

/// Decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Minimum confidence threshold
    pub confidence_threshold: f64,
    /// Time window for spike counting (ms)
    pub time_window: f64,
    /// Number of output neurons
    pub output_size: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            time_window: 10.0,
            output_size: 3,  // BUY, SELL, HOLD
        }
    }
}

/// Trait for spike decoders
pub trait Decoder: Send + Sync {
    /// Decode output spikes into a trading signal
    fn decode(&self, spikes: &[SpikeEvent]) -> TradingSignal;

    /// Reset decoder state
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal_buy() {
        let signal = TradingSignal::Buy {
            confidence: 0.8,
            urgency: 0.5,
        };
        assert!(signal.is_buy());
        assert!(!signal.is_sell());
        assert!(!signal.is_hold());
        assert_eq!(signal.confidence(), Some(0.8));
    }

    #[test]
    fn test_trading_signal_hold() {
        let signal = TradingSignal::Hold;
        assert!(signal.is_hold());
        assert_eq!(signal.confidence(), None);
    }
}
