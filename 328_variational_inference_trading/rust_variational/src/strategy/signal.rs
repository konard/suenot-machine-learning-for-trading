//! Trading signal generation

use crate::variational::prediction::Prediction;
use serde::{Deserialize, Serialize};

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Long,
    Short,
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Long => write!(f, "LONG"),
            SignalType::Short => write!(f, "SHORT"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// Trading signal with confidence and uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Expected return
    pub expected_return: f64,
    /// Return uncertainty (standard deviation)
    pub uncertainty: f64,
    /// Probability of positive return
    pub prob_positive: f64,
    /// Detected market regime
    pub regime: usize,
    /// Position size recommendation (as fraction of portfolio)
    pub position_size: f64,
    /// Optional timestamp
    pub timestamp: Option<i64>,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        signal_type: SignalType,
        confidence: f64,
        expected_return: f64,
        uncertainty: f64,
        prob_positive: f64,
        regime: usize,
    ) -> Self {
        Self {
            signal_type,
            confidence,
            expected_return,
            uncertainty,
            prob_positive,
            regime,
            position_size: 0.0,
            timestamp: None,
        }
    }

    /// Set position size
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size;
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, ts: i64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Check if signal is actionable (not HOLD)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }
}

/// Signal generator
pub struct SignalGenerator {
    /// Confidence threshold for generating signals
    confidence_threshold: f64,
    /// Uncertainty threshold (max allowed uncertainty)
    uncertainty_threshold: f64,
    /// Maximum position size
    max_position_size: f64,
    /// Minimum position size
    min_position_size: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.6)
    }
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
            uncertainty_threshold: 0.05,
            max_position_size: 0.1,
            min_position_size: 0.01,
        }
    }

    /// Configure uncertainty threshold
    pub fn with_uncertainty_threshold(mut self, threshold: f64) -> Self {
        self.uncertainty_threshold = threshold;
        self
    }

    /// Configure position limits
    pub fn with_position_limits(mut self, min: f64, max: f64) -> Self {
        self.min_position_size = min;
        self.max_position_size = max;
        self
    }

    /// Generate signal from prediction
    pub fn generate(&self, prediction: &Prediction) -> Signal {
        // Check uncertainty
        if prediction.return_std > self.uncertainty_threshold {
            return Signal::new(
                SignalType::Hold,
                0.5,
                prediction.expected_return,
                prediction.return_std,
                prediction.prob_positive,
                prediction.regime,
            );
        }

        // Determine signal type based on probability
        let (signal_type, confidence) = if prediction.prob_positive > self.confidence_threshold {
            (SignalType::Long, prediction.prob_positive)
        } else if prediction.prob_positive < (1.0 - self.confidence_threshold) {
            (SignalType::Short, 1.0 - prediction.prob_positive)
        } else {
            (SignalType::Hold, 0.5)
        };

        // Calculate position size
        let position_size = self.calculate_position_size(confidence, prediction.return_std);

        Signal::new(
            signal_type,
            confidence,
            prediction.expected_return,
            prediction.return_std,
            prediction.prob_positive,
            prediction.regime,
        ).with_position_size(position_size)
    }

    /// Calculate position size based on confidence and uncertainty
    fn calculate_position_size(&self, confidence: f64, uncertainty: f64) -> f64 {
        if confidence < self.confidence_threshold {
            return 0.0;
        }

        // Base size from confidence
        let base_size = confidence * self.max_position_size;

        // Adjust for uncertainty
        let uncertainty_factor = (1.0 - uncertainty / self.uncertainty_threshold).max(0.1);
        let adjusted_size = base_size * uncertainty_factor;

        // Clamp to limits
        adjusted_size.clamp(self.min_position_size, self.max_position_size)
    }

    /// Generate signals for multiple predictions
    pub fn generate_batch(&self, predictions: &[Prediction]) -> Vec<Signal> {
        predictions.iter().map(|p| self.generate(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new(0.6);

        // High confidence bullish prediction
        let pred = Prediction::new(
            0.02,
            0.01,
            0,
            vec![0.8, 0.1, 0.1],
            vec![],
            vec![],
        );

        let signal = generator.generate(&pred);
        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.confidence > 0.6);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_hold_on_high_uncertainty() {
        let generator = SignalGenerator::new(0.6)
            .with_uncertainty_threshold(0.02);

        // High uncertainty prediction
        let mut pred = Prediction::new(
            0.02,
            0.05,  // High uncertainty
            0,
            vec![0.8, 0.1, 0.1],
            vec![],
            vec![],
        );
        pred.prob_positive = 0.9;  // Would be long otherwise

        let signal = generator.generate(&pred);
        assert_eq!(signal.signal_type, SignalType::Hold);
    }

    #[test]
    fn test_short_signal() {
        let generator = SignalGenerator::new(0.6);

        let mut pred = Prediction::new(
            -0.02,
            0.01,
            1,
            vec![0.1, 0.8, 0.1],
            vec![],
            vec![],
        );
        pred.prob_positive = 0.2;  // Bearish

        let signal = generator.generate(&pred);
        assert_eq!(signal.signal_type, SignalType::Short);
    }
}
