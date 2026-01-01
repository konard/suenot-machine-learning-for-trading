//! Trading signal types and generation

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Confidence threshold for generating signals
pub const CONFIDENCE_THRESHOLD: f64 = 0.55;

/// Trading signal
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    /// Long position (buy)
    Long {
        /// Confidence level (0.0 to 1.0)
        confidence: f64,
    },
    /// Short position (sell)
    Short {
        /// Confidence level (0.0 to 1.0)
        confidence: f64,
    },
    /// No position
    Hold,
}

impl Signal {
    /// Create signal from model output probabilities
    ///
    /// # Arguments
    /// * `output` - Model output probabilities [batch_size, 3] (long, short, hold)
    pub fn from_output(output: &Array2<f64>) -> Self {
        // Get first sample in batch
        let long_prob = output[[0, 0]];
        let short_prob = output[[0, 1]];
        let _hold_prob = output[[0, 2]];

        if long_prob > CONFIDENCE_THRESHOLD && long_prob > short_prob {
            Signal::Long {
                confidence: long_prob,
            }
        } else if short_prob > CONFIDENCE_THRESHOLD && short_prob > long_prob {
            Signal::Short {
                confidence: short_prob,
            }
        } else {
            Signal::Hold
        }
    }

    /// Create signal from probabilities
    pub fn from_probs(long: f64, short: f64, hold: f64) -> Self {
        if long > CONFIDENCE_THRESHOLD && long > short && long > hold {
            Signal::Long { confidence: long }
        } else if short > CONFIDENCE_THRESHOLD && short > long && short > hold {
            Signal::Short { confidence: short }
        } else {
            Signal::Hold
        }
    }

    /// Check if signal is long
    pub fn is_long(&self) -> bool {
        matches!(self, Signal::Long { .. })
    }

    /// Check if signal is short
    pub fn is_short(&self) -> bool {
        matches!(self, Signal::Short { .. })
    }

    /// Check if signal is hold
    pub fn is_hold(&self) -> bool {
        matches!(self, Signal::Hold)
    }

    /// Get confidence level
    pub fn confidence(&self) -> f64 {
        match self {
            Signal::Long { confidence } => *confidence,
            Signal::Short { confidence } => *confidence,
            Signal::Hold => 0.0,
        }
    }

    /// Get signal direction as number (-1, 0, 1)
    pub fn direction(&self) -> i32 {
        match self {
            Signal::Long { .. } => 1,
            Signal::Short { .. } => -1,
            Signal::Hold => 0,
        }
    }

    /// Get signal strength (confidence-adjusted direction)
    pub fn strength(&self) -> f64 {
        match self {
            Signal::Long { confidence } => *confidence,
            Signal::Short { confidence } => -*confidence,
            Signal::Hold => 0.0,
        }
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Long { confidence } => write!(f, "LONG ({:.1}%)", confidence * 100.0),
            Signal::Short { confidence } => write!(f, "SHORT ({:.1}%)", confidence * 100.0),
            Signal::Hold => write!(f, "HOLD"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_signal_from_output() {
        // Strong long signal
        let output = Array2::from_shape_vec((1, 3), vec![0.7, 0.2, 0.1]).unwrap();
        let signal = Signal::from_output(&output);
        assert!(signal.is_long());
        assert!((signal.confidence() - 0.7).abs() < 1e-6);

        // Strong short signal
        let output = Array2::from_shape_vec((1, 3), vec![0.1, 0.8, 0.1]).unwrap();
        let signal = Signal::from_output(&output);
        assert!(signal.is_short());

        // Uncertain - hold
        let output = Array2::from_shape_vec((1, 3), vec![0.4, 0.3, 0.3]).unwrap();
        let signal = Signal::from_output(&output);
        assert!(signal.is_hold());
    }

    #[test]
    fn test_signal_direction() {
        assert_eq!(Signal::Long { confidence: 0.7 }.direction(), 1);
        assert_eq!(Signal::Short { confidence: 0.8 }.direction(), -1);
        assert_eq!(Signal::Hold.direction(), 0);
    }

    #[test]
    fn test_signal_display() {
        let signal = Signal::Long { confidence: 0.75 };
        assert_eq!(format!("{}", signal), "LONG (75.0%)");
    }
}
