//! Trading strategy with attention-based confidence filtering.

use super::Signal;
use crate::model::AttentionWeights;

/// Configuration for the trading strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Minimum direction probability for long signal
    pub direction_threshold_long: f64,
    /// Maximum direction probability for short signal
    pub direction_threshold_short: f64,
    /// Minimum absolute predicted return
    pub return_threshold: f64,
    /// Minimum attention confidence to trade
    pub confidence_threshold: f64,
    /// Position size as fraction of portfolio
    pub position_size: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            direction_threshold_long: 0.6,
            direction_threshold_short: 0.4,
            return_threshold: 0.001,
            confidence_threshold: 0.5,
            position_size: 1.0,
        }
    }
}

/// Trading strategy with attention-based confidence filtering.
#[derive(Debug, Clone)]
pub struct TradingStrategy {
    config: StrategyConfig,
}

impl TradingStrategy {
    /// Create a new trading strategy with default configuration.
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
        }
    }

    /// Create a new trading strategy with custom configuration.
    pub fn with_config(config: StrategyConfig) -> Self {
        Self { config }
    }

    /// Generate trading signal based on predictions and attention.
    ///
    /// # Arguments
    ///
    /// * `pred_return` - Predicted return
    /// * `pred_direction` - Direction probability (>0.5 = up)
    /// * `attention` - Optional attention weights for confidence
    /// * `seq_len` - Sequence length (for confidence normalization)
    ///
    /// # Returns
    ///
    /// Tuple of (signal, confidence)
    pub fn generate_signal(
        &self,
        pred_return: f64,
        pred_direction: f64,
        attention: Option<&AttentionWeights>,
        seq_len: usize,
    ) -> (Signal, f64) {
        // Compute attention-based confidence
        let confidence = attention
            .map(|a| a.compute_confidence(seq_len))
            .unwrap_or(1.0);

        // Check confidence threshold
        if confidence < self.config.confidence_threshold {
            return (Signal::Hold, confidence);
        }

        // Generate signal based on predictions
        if pred_direction > self.config.direction_threshold_long
            && pred_return > self.config.return_threshold
        {
            (Signal::Long, confidence)
        } else if pred_direction < self.config.direction_threshold_short
            && pred_return < -self.config.return_threshold
        {
            (Signal::Short, confidence)
        } else {
            (Signal::Hold, confidence)
        }
    }

    /// Get the position size.
    pub fn position_size(&self) -> f64 {
        self.config.position_size
    }

    /// Get the configuration.
    pub fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_signal() {
        let strategy = TradingStrategy::new();
        let (signal, _) = strategy.generate_signal(0.01, 0.7, None, 60);
        assert_eq!(signal, Signal::Long);
    }

    #[test]
    fn test_short_signal() {
        let strategy = TradingStrategy::new();
        let (signal, _) = strategy.generate_signal(-0.01, 0.3, None, 60);
        assert_eq!(signal, Signal::Short);
    }

    #[test]
    fn test_hold_signal() {
        let strategy = TradingStrategy::new();
        let (signal, _) = strategy.generate_signal(0.0001, 0.5, None, 60);
        assert_eq!(signal, Signal::Hold);
    }
}
