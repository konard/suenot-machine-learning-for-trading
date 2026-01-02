//! Trading signal generation based on associative memory
//!
//! Converts memory predictions to actionable trading signals

use crate::memory::{DenseAssociativeMemory, PatternMemoryManager};
use ndarray::Array1;

/// Trading signal
#[derive(Debug, Clone, Copy)]
pub struct Signal {
    /// Direction: positive = long, negative = short, zero = neutral
    pub direction: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Suggested position size (0-1)
    pub position_size: f64,
    /// Whether this is a high-confidence signal
    pub is_actionable: bool,
}

impl Signal {
    /// Create a new signal
    pub fn new(direction: f64, confidence: f64) -> Self {
        Self {
            direction,
            confidence,
            position_size: 0.0,
            is_actionable: false,
        }
    }

    /// Create a neutral (no trade) signal
    pub fn neutral() -> Self {
        Self {
            direction: 0.0,
            confidence: 0.0,
            position_size: 0.0,
            is_actionable: false,
        }
    }

    /// Check if signal suggests going long
    pub fn is_long(&self) -> bool {
        self.is_actionable && self.direction > 0.0
    }

    /// Check if signal suggests going short
    pub fn is_short(&self) -> bool {
        self.is_actionable && self.direction < 0.0
    }
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Minimum confidence to generate a signal
    pub min_confidence: f64,
    /// Minimum prediction magnitude
    pub min_prediction: f64,
    /// Maximum position size
    pub max_position: f64,
    /// Position sizing mode
    pub sizing_mode: PositionSizingMode,
    /// Number of patterns to retrieve
    pub k_patterns: usize,
    /// Consensus threshold (percentage of patterns agreeing)
    pub consensus_threshold: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            min_prediction: 0.001, // 0.1%
            max_position: 1.0,
            sizing_mode: PositionSizingMode::Linear,
            k_patterns: 5,
            consensus_threshold: 0.6,
        }
    }
}

/// Position sizing modes
#[derive(Debug, Clone, Copy)]
pub enum PositionSizingMode {
    /// Fixed size regardless of confidence
    Fixed,
    /// Linear scaling with confidence
    Linear,
    /// Quadratic scaling (more conservative)
    Quadratic,
    /// Kelly criterion based
    Kelly,
}

/// Signal generator using associative memory
pub struct SignalGenerator {
    config: SignalConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate signal from Dense Associative Memory prediction
    pub fn generate_from_dam(
        &self,
        memory: &DenseAssociativeMemory,
        pattern: &Array1<f64>,
    ) -> Signal {
        let (prediction, confidence) = memory.predict(pattern);

        self.create_signal(prediction, confidence)
    }

    /// Generate signal from Pattern Memory Manager
    pub fn generate_from_manager(
        &self,
        manager: &mut PatternMemoryManager,
        pattern: &[f64],
    ) -> Signal {
        let (_, outcomes, similarities) = manager.query(pattern, self.config.k_patterns);

        if outcomes.is_empty() {
            return Signal::neutral();
        }

        // Weighted prediction
        let total_weight: f64 = similarities.iter().sum();
        let prediction = if total_weight > 0.0 {
            outcomes
                .iter()
                .zip(similarities.iter())
                .map(|(o, s)| o * s)
                .sum::<f64>()
                / total_weight
        } else {
            0.0
        };

        // Confidence from similarity
        let confidence = similarities.iter().sum::<f64>() / similarities.len() as f64;

        // Check consensus
        let positive_count = outcomes.iter().filter(|&&o| o > 0.0).count();
        let consensus = positive_count as f64 / outcomes.len() as f64;
        let has_consensus = consensus >= self.config.consensus_threshold
            || (1.0 - consensus) >= self.config.consensus_threshold;

        let mut signal = self.create_signal(prediction, confidence);

        // Require consensus for actionable signals
        if !has_consensus {
            signal.is_actionable = false;
            signal.position_size = 0.0;
        }

        signal
    }

    /// Create a signal from prediction and confidence
    fn create_signal(&self, prediction: f64, confidence: f64) -> Signal {
        let mut signal = Signal::new(prediction.signum(), confidence);

        // Check if actionable
        if confidence < self.config.min_confidence {
            return Signal::neutral();
        }

        if prediction.abs() < self.config.min_prediction {
            return Signal::neutral();
        }

        signal.is_actionable = true;

        // Calculate position size
        signal.position_size = self.calculate_position_size(prediction, confidence);

        signal
    }

    /// Calculate position size based on prediction and confidence
    fn calculate_position_size(&self, prediction: f64, confidence: f64) -> f64 {
        let base_size = match self.config.sizing_mode {
            PositionSizingMode::Fixed => self.config.max_position,

            PositionSizingMode::Linear => confidence * self.config.max_position,

            PositionSizingMode::Quadratic => confidence.powi(2) * self.config.max_position,

            PositionSizingMode::Kelly => {
                // Simplified Kelly: edge / odds
                // Assuming symmetric returns, edge ~ prediction magnitude
                let edge = prediction.abs().min(0.1); // Cap edge at 10%
                let kelly_fraction = edge * confidence;
                (kelly_fraction * 0.25).min(self.config.max_position) // Use quarter Kelly
            }
        };

        base_size.min(self.config.max_position)
    }
}

/// Extended signal with retrieval details
#[derive(Debug, Clone)]
pub struct DetailedSignal {
    /// Base signal
    pub signal: Signal,
    /// Prediction value
    pub prediction: f64,
    /// Similar patterns (index, similarity, outcome)
    pub similar_patterns: Vec<(usize, f64, f64)>,
    /// Average similarity
    pub avg_similarity: f64,
    /// Consensus (percentage agreeing with direction)
    pub consensus: f64,
}

impl DetailedSignal {
    /// Create from memory retrieval
    pub fn from_retrieval(
        prediction: f64,
        confidence: f64,
        similar_patterns: Vec<(usize, f64, f64)>,
        config: &SignalConfig,
    ) -> Self {
        let avg_similarity = if similar_patterns.is_empty() {
            0.0
        } else {
            similar_patterns.iter().map(|(_, s, _)| s).sum::<f64>() / similar_patterns.len() as f64
        };

        let consensus = if similar_patterns.is_empty() {
            0.5
        } else {
            let agreeing = similar_patterns
                .iter()
                .filter(|(_, _, o)| o.signum() == prediction.signum())
                .count();
            agreeing as f64 / similar_patterns.len() as f64
        };

        let mut signal = Signal::new(prediction.signum(), confidence);

        // Determine if actionable
        signal.is_actionable = confidence >= config.min_confidence
            && prediction.abs() >= config.min_prediction
            && (consensus >= config.consensus_threshold
                || (1.0 - consensus) >= config.consensus_threshold);

        if signal.is_actionable {
            let generator = SignalGenerator::new(config.clone());
            signal.position_size = generator.calculate_position_size(prediction, confidence);
        }

        Self {
            signal,
            prediction,
            similar_patterns,
            avg_similarity,
            consensus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let config = SignalConfig::default();
        let generator = SignalGenerator::new(config);

        let signal = generator.create_signal(0.02, 0.5);

        assert!(signal.is_actionable);
        assert!(signal.is_long());
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_low_confidence_neutral() {
        let config = SignalConfig {
            min_confidence: 0.5,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let signal = generator.create_signal(0.02, 0.3);

        assert!(!signal.is_actionable);
    }

    #[test]
    fn test_position_sizing() {
        let config = SignalConfig {
            sizing_mode: PositionSizingMode::Linear,
            max_position: 1.0,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let size1 = generator.calculate_position_size(0.02, 0.5);
        let size2 = generator.calculate_position_size(0.02, 1.0);

        assert!(size2 > size1);
    }
}
