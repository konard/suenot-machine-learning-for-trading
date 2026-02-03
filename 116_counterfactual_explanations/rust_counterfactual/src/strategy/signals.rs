//! Trading signal generation.

use crate::model::TradingClassifier;

/// Generate trading signals from model predictions.
///
/// # Arguments
///
/// * `model` - Trained classifier
/// * `features` - Feature matrix
/// * `confidence_threshold` - Minimum confidence for non-HOLD signals
///
/// # Returns
///
/// Vector of signals (0=SELL, 1=HOLD, 2=BUY)
pub fn generate_signals(
    model: &TradingClassifier,
    features: &[Vec<f64>],
    confidence_threshold: f64,
) -> Vec<usize> {
    features
        .iter()
        .map(|x| {
            let probs = model.predict_proba(x);
            let pred = model.predict(x);
            let confidence = probs[pred];

            if confidence < confidence_threshold {
                1 // HOLD
            } else {
                pred
            }
        })
        .collect()
}

/// Signal with confidence information.
#[derive(Debug, Clone)]
pub struct SignalInfo {
    /// Signal type (0=SELL, 1=HOLD, 2=BUY)
    pub signal: usize,
    /// Confidence level
    pub confidence: f64,
    /// Probability distribution
    pub probabilities: Vec<f64>,
}

impl SignalInfo {
    /// Get signal name.
    pub fn name(&self) -> &'static str {
        match self.signal {
            0 => "SELL",
            1 => "HOLD",
            2 => "BUY",
            _ => "UNKNOWN",
        }
    }
}

/// Generate detailed signal information.
pub fn generate_signals_with_info(
    model: &TradingClassifier,
    features: &[Vec<f64>],
) -> Vec<SignalInfo> {
    features
        .iter()
        .map(|x| {
            let probs = model.predict_proba(x);
            let pred = model.predict(x);
            let confidence = probs[pred];

            SignalInfo {
                signal: pred,
                confidence,
                probabilities: probs,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_signals() {
        let model = TradingClassifier::new(6, 64, 3);
        let features = vec![
            vec![0.5, 0.01, -0.2, 0.02, 0.3, 0.01],
            vec![-0.5, -0.01, 0.2, 0.02, -0.3, -0.01],
        ];
        let signals = generate_signals(&model, &features, 0.5);
        assert_eq!(signals.len(), 2);
    }
}
