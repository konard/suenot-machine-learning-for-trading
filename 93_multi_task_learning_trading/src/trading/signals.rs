//! Trading signal generation from multi-task model predictions.

use crate::model::network::TaskPredictions;

/// Trading signal derived from multi-task predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Go long (buy)
    Long,
    /// Go short (sell)
    Short,
    /// Stay neutral (no position)
    Neutral,
}

impl TradingSignal {
    /// Convert to position multiplier: Long=1, Short=-1, Neutral=0
    pub fn position(&self) -> f64 {
        match self {
            TradingSignal::Long => 1.0,
            TradingSignal::Short => -1.0,
            TradingSignal::Neutral => 0.0,
        }
    }
}

/// Signal generator that combines multi-task predictions into trading signals.
pub struct SignalGenerator {
    return_threshold: f64,
    direction_threshold: f64,
    volatility_cap: f64,
}

impl SignalGenerator {
    /// Create a new signal generator.
    ///
    /// # Arguments
    /// * `return_threshold` - Minimum predicted return to trigger a trade
    /// * `direction_threshold` - Minimum direction confidence (0.5 = no edge)
    /// * `volatility_cap` - Maximum volatility to allow trading
    pub fn new(return_threshold: f64, direction_threshold: f64, volatility_cap: f64) -> Self {
        Self {
            return_threshold,
            direction_threshold,
            volatility_cap,
        }
    }

    /// Generate a trading signal from multi-task predictions.
    pub fn generate(&self, predictions: &TaskPredictions, index: usize) -> TradingSignal {
        let ret = predictions.return_pred.get(index).copied().unwrap_or(0.0);
        let dir = predictions.direction_pred.get(index).copied().unwrap_or(0.5);
        let vol = predictions.volatility_pred.get(index).copied().unwrap_or(0.0);

        // Skip if volatility is too high
        if vol > self.volatility_cap {
            return TradingSignal::Neutral;
        }

        // Combine return magnitude with direction confidence
        let signal_strength = ret * (2.0 * dir - 1.0);

        if signal_strength > self.return_threshold && dir > self.direction_threshold {
            TradingSignal::Long
        } else if signal_strength < -self.return_threshold && dir < (1.0 - self.direction_threshold) {
            TradingSignal::Short
        } else {
            TradingSignal::Neutral
        }
    }

    /// Generate signals for all samples in a batch prediction.
    pub fn generate_batch(&self, predictions: &TaskPredictions) -> Vec<TradingSignal> {
        (0..predictions.return_pred.len())
            .map(|i| self.generate(predictions, i))
            .collect()
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.001, 0.55, 0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let gen = SignalGenerator::new(0.001, 0.55, 0.05);

        let preds = TaskPredictions {
            return_pred: vec![0.02],
            direction_pred: vec![0.8],
            volatility_pred: vec![0.01],
            volume_pred: vec![1.0],
        };
        assert_eq!(gen.generate(&preds, 0), TradingSignal::Long);

        let preds = TaskPredictions {
            return_pred: vec![-0.02],
            direction_pred: vec![0.2],
            volatility_pred: vec![0.01],
            volume_pred: vec![1.0],
        };
        assert_eq!(gen.generate(&preds, 0), TradingSignal::Short);
    }

    #[test]
    fn test_high_volatility_filter() {
        let gen = SignalGenerator::new(0.001, 0.55, 0.05);

        let preds = TaskPredictions {
            return_pred: vec![0.05],
            direction_pred: vec![0.9],
            volatility_pred: vec![0.10], // Too high
            volume_pred: vec![1.0],
        };
        assert_eq!(gen.generate(&preds, 0), TradingSignal::Neutral);
    }

    #[test]
    fn test_position_values() {
        assert_eq!(TradingSignal::Long.position(), 1.0);
        assert_eq!(TradingSignal::Short.position(), -1.0);
        assert_eq!(TradingSignal::Neutral.position(), 0.0);
    }
}
