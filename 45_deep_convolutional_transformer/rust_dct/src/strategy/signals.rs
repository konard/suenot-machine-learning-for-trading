//! Signal generation from model predictions

use crate::model::{DCTModel, Prediction};
use ndarray::Array3;

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// No position
    Hold,
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// The signal type
    pub signal: Signal,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted class (0=Up, 1=Down, 2=Stable)
    pub predicted_class: usize,
    /// Class probabilities
    pub probabilities: Vec<f64>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(prediction: &Prediction, confidence_threshold: f64) -> Self {
        let signal = if prediction.confidence < confidence_threshold {
            Signal::Hold
        } else {
            match prediction.predicted_class {
                0 => Signal::Long,  // Up prediction -> Buy
                1 => Signal::Short, // Down prediction -> Sell
                _ => Signal::Hold,  // Stable or uncertain
            }
        };

        Self {
            signal,
            confidence: prediction.confidence,
            predicted_class: prediction.predicted_class,
            probabilities: prediction.probabilities.clone(),
        }
    }

    /// Check if this is an actionable signal
    pub fn is_actionable(&self) -> bool {
        self.signal != Signal::Hold
    }
}

/// Generate trading signals from model
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Minimum confidence to generate a signal
    pub confidence_threshold: f64,
    /// Minimum probability difference for signal
    pub min_prob_diff: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_prob_diff: 0.1,
        }
    }
}

impl SignalGenerator {
    /// Create new signal generator with custom threshold
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
            ..Default::default()
        }
    }

    /// Generate signals from model predictions
    pub fn generate(&self, predictions: &[Prediction]) -> Vec<TradingSignal> {
        predictions
            .iter()
            .map(|pred| {
                let mut signal = TradingSignal::new(pred, self.confidence_threshold);

                // Additional filter: check probability difference
                if signal.is_actionable() && pred.probabilities.len() >= 2 {
                    let sorted: Vec<f64> = {
                        let mut p: Vec<f64> = pred.probabilities.clone();
                        p.sort_by(|a: &f64, b: &f64| b.partial_cmp(a).unwrap());
                        p
                    };

                    if sorted[0] - sorted[1] < self.min_prob_diff {
                        signal.signal = Signal::Hold;
                    }
                }

                signal
            })
            .collect()
    }

    /// Generate signal for single input
    pub fn generate_single(&self, model: &DCTModel, x: &Array3<f64>) -> TradingSignal {
        let predictions = model.predict(x);
        self.generate(&predictions).into_iter().next().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let predictions = vec![
            Prediction {
                probabilities: vec![0.7, 0.2, 0.1],
                predicted_class: 0,
                confidence: 0.7,
            },
            Prediction {
                probabilities: vec![0.2, 0.6, 0.2],
                predicted_class: 1,
                confidence: 0.6,
            },
            Prediction {
                probabilities: vec![0.3, 0.3, 0.4],
                predicted_class: 2,
                confidence: 0.4,
            },
        ];

        let generator = SignalGenerator::new(0.5);
        let signals = generator.generate(&predictions);

        assert_eq!(signals.len(), 3);
        assert_eq!(signals[0].signal, Signal::Long);
        assert_eq!(signals[1].signal, Signal::Short);
        assert_eq!(signals[2].signal, Signal::Hold);
    }

    #[test]
    fn test_low_confidence_becomes_hold() {
        let prediction = Prediction {
            probabilities: vec![0.4, 0.35, 0.25],
            predicted_class: 0,
            confidence: 0.4,
        };

        let generator = SignalGenerator::new(0.5);
        let signals = generator.generate(&[prediction]);

        assert_eq!(signals[0].signal, Signal::Hold);
    }
}
