//! Trading signal generation with uncertainty

use super::config::SignalConfig;
use crate::model::prediction::PredictionResult;
use serde::{Deserialize, Serialize};

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Go long (buy)
    Long,
    /// Go short (sell)
    Short,
    /// Hold (no action)
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

/// Trading signal with uncertainty information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal type (Long, Short, Hold)
    pub signal_type: SignalType,
    /// Predicted return
    pub predicted_return: f64,
    /// Uncertainty (standard deviation)
    pub uncertainty: f64,
    /// Confidence (z-score)
    pub confidence: f64,
    /// Position size (fraction of portfolio)
    pub position_size: f64,
    /// Reason for the signal
    pub reason: String,
    /// Timestamp (if available)
    pub timestamp: Option<i64>,
}

impl Signal {
    /// Create a hold signal
    pub fn hold(reason: &str) -> Self {
        Self {
            signal_type: SignalType::Hold,
            predicted_return: 0.0,
            uncertainty: 0.0,
            confidence: 0.0,
            position_size: 0.0,
            reason: reason.to_string(),
            timestamp: None,
        }
    }

    /// Check if signal is actionable (not hold)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }
}

/// Signal generator with uncertainty-based filtering
pub struct SignalGenerator {
    /// Configuration
    pub config: SignalConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate trading signal from prediction result
    pub fn generate(&self, prediction: &PredictionResult) -> Signal {
        let mean = prediction.mean;
        let std = prediction.std;

        // Calculate z-score
        let z_score = if std > 0.0 { mean.abs() / std } else { 0.0 };

        // Filter by uncertainty
        if std > self.config.uncertainty_threshold {
            return Signal {
                signal_type: SignalType::Hold,
                predicted_return: mean,
                uncertainty: std,
                confidence: z_score,
                position_size: 0.0,
                reason: format!(
                    "Uncertainty too high: {:.4} > {:.4}",
                    std, self.config.uncertainty_threshold
                ),
                timestamp: None,
            };
        }

        // Filter by confidence
        if z_score < self.config.confidence_threshold {
            return Signal {
                signal_type: SignalType::Hold,
                predicted_return: mean,
                uncertainty: std,
                confidence: z_score,
                position_size: 0.0,
                reason: format!(
                    "Confidence too low: {:.2} < {:.2}",
                    z_score, self.config.confidence_threshold
                ),
                timestamp: None,
            };
        }

        // Filter by minimum return
        if mean.abs() < self.config.min_return_threshold {
            return Signal {
                signal_type: SignalType::Hold,
                predicted_return: mean,
                uncertainty: std,
                confidence: z_score,
                position_size: 0.0,
                reason: format!(
                    "Expected return too small: {:.6}",
                    mean
                ),
                timestamp: None,
            };
        }

        // Determine direction
        let signal_type = if mean > 0.0 {
            SignalType::Long
        } else {
            SignalType::Short
        };

        // Calculate position size
        let position_size = if self.config.use_kelly {
            self.kelly_position_size(mean.abs(), std)
        } else {
            self.simple_position_size(z_score, std)
        };

        Signal {
            signal_type,
            predicted_return: mean,
            uncertainty: std,
            confidence: z_score,
            position_size,
            reason: "Signal generated".to_string(),
            timestamp: None,
        }
    }

    /// Calculate position size using Kelly criterion
    fn kelly_position_size(&self, expected_return: f64, uncertainty: f64) -> f64 {
        if uncertainty <= 0.0 {
            return 0.0;
        }

        // Kelly fraction: f = mu / sigma^2
        let kelly = expected_return / (uncertainty * uncertainty);

        // Half Kelly for safety
        let half_kelly = kelly * 0.5;

        // Apply uncertainty penalty
        let confidence_factor = 1.0 / (1.0 + uncertainty);
        let adjusted = half_kelly * confidence_factor;

        // Clip to allowed range
        adjusted
            .max(self.config.min_position_size)
            .min(self.config.max_position_size)
            .max(0.0)
    }

    /// Calculate position size based on confidence
    fn simple_position_size(&self, confidence: f64, uncertainty: f64) -> f64 {
        // Base size from confidence
        let base_size = (confidence / 5.0).min(1.0);

        // Reduce based on uncertainty
        let uncertainty_penalty = 1.0 - (uncertainty / self.config.uncertainty_threshold).min(1.0);

        let size = base_size * uncertainty_penalty * self.config.max_position_size;

        size
            .max(self.config.min_position_size)
            .min(self.config.max_position_size)
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_uncertainty_hold() {
        let config = SignalConfig {
            uncertainty_threshold: 0.02,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let prediction = PredictionResult::from_samples(vec![0.01, 0.02, 0.03, -0.02, 0.05]);
        // This has high uncertainty
        let signal = generator.generate(&prediction);

        assert_eq!(signal.signal_type, SignalType::Hold);
        assert!(signal.reason.contains("Uncertainty"));
    }

    #[test]
    fn test_low_confidence_hold() {
        let config = SignalConfig {
            confidence_threshold: 3.0,
            uncertainty_threshold: 0.1,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let prediction = PredictionResult {
            mean: 0.01,
            std: 0.02,
            variance: 0.0004,
            samples: vec![],
            n_samples: 50,
        };
        let signal = generator.generate(&prediction);

        assert_eq!(signal.signal_type, SignalType::Hold);
        assert!(signal.reason.contains("Confidence"));
    }

    #[test]
    fn test_confident_long_signal() {
        let config = SignalConfig {
            confidence_threshold: 1.0,
            uncertainty_threshold: 0.05,
            min_return_threshold: 0.001,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let prediction = PredictionResult {
            mean: 0.02,
            std: 0.01,
            variance: 0.0001,
            samples: vec![],
            n_samples: 50,
        };
        let signal = generator.generate(&prediction);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_confident_short_signal() {
        let config = SignalConfig {
            confidence_threshold: 1.0,
            uncertainty_threshold: 0.05,
            min_return_threshold: 0.001,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let prediction = PredictionResult {
            mean: -0.02,
            std: 0.01,
            variance: 0.0001,
            samples: vec![],
            n_samples: 50,
        };
        let signal = generator.generate(&prediction);

        assert_eq!(signal.signal_type, SignalType::Short);
        assert!(signal.position_size > 0.0);
    }
}
