//! Trading signal generation from QuantNet model predictions.

use crate::model::network::{QuantNetPredictions, TradingOutput};

/// Trading signal derived from QuantNet predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Go long (buy)
    Buy,
    /// Go short (sell)
    Sell,
    /// Stay flat (no position)
    Hold,
}

impl TradingSignal {
    /// Convert to position multiplier: Buy=1, Sell=-1, Hold=0
    pub fn position(&self) -> f64 {
        match self {
            TradingSignal::Buy => 1.0,
            TradingSignal::Sell => -1.0,
            TradingSignal::Hold => 0.0,
        }
    }
}

/// Signal generator that converts QuantNet outputs into trading signals.
pub struct SignalGenerator {
    signal_threshold: f64,
    confidence_threshold: f64,
}

impl SignalGenerator {
    /// Create a new signal generator.
    ///
    /// # Arguments
    /// * `signal_threshold` - Minimum absolute signal strength to trigger a trade
    /// * `confidence_threshold` - Minimum confidence to allow trading
    pub fn new(signal_threshold: f64, confidence_threshold: f64) -> Self {
        Self {
            signal_threshold,
            confidence_threshold,
        }
    }

    /// Generate a trading signal from a single asset's trading output.
    pub fn generate_from_output(&self, output: &TradingOutput) -> TradingSignal {
        // Skip if confidence is too low
        if output.confidence < self.confidence_threshold {
            return TradingSignal::Hold;
        }

        if output.signal > self.signal_threshold {
            TradingSignal::Buy
        } else if output.signal < -self.signal_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Generate signals for all assets from QuantNet predictions.
    pub fn generate_all(&self, predictions: &QuantNetPredictions) -> Vec<(String, TradingSignal)> {
        predictions.asset_outputs
            .iter()
            .map(|(name, output)| {
                (name.clone(), self.generate_from_output(output))
            })
            .collect()
    }

    /// Generate a signal for a specific asset by index.
    pub fn generate_for_asset(
        &self,
        predictions: &QuantNetPredictions,
        asset_index: usize,
    ) -> TradingSignal {
        predictions.asset_outputs
            .get(asset_index)
            .map(|(_, output)| self.generate_from_output(output))
            .unwrap_or(TradingSignal::Hold)
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.1, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_output() {
        let gen = SignalGenerator::new(0.1, 0.5);

        let buy_output = TradingOutput { signal: 0.5, return_pred: 0.02, confidence: 0.8 };
        assert_eq!(gen.generate_from_output(&buy_output), TradingSignal::Buy);

        let sell_output = TradingOutput { signal: -0.5, return_pred: -0.02, confidence: 0.8 };
        assert_eq!(gen.generate_from_output(&sell_output), TradingSignal::Sell);

        let hold_output = TradingOutput { signal: 0.05, return_pred: 0.001, confidence: 0.8 };
        assert_eq!(gen.generate_from_output(&hold_output), TradingSignal::Hold);
    }

    #[test]
    fn test_low_confidence_filter() {
        let gen = SignalGenerator::new(0.1, 0.5);

        let output = TradingOutput { signal: 0.9, return_pred: 0.05, confidence: 0.3 };
        assert_eq!(gen.generate_from_output(&output), TradingSignal::Hold);
    }

    #[test]
    fn test_position_values() {
        assert_eq!(TradingSignal::Buy.position(), 1.0);
        assert_eq!(TradingSignal::Sell.position(), -1.0);
        assert_eq!(TradingSignal::Hold.position(), 0.0);
    }
}
