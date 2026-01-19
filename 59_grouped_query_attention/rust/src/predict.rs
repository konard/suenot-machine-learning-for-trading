//! Prediction Utilities
//!
//! This module provides functions for making predictions with trained GQA models.

use ndarray::{Array1, Array2};

use crate::model::GQATrader;

/// Prediction result with probabilities
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted class (0=down, 1=neutral, 2=up)
    pub prediction: usize,
    /// Class probabilities
    pub probabilities: [f32; 3],
    /// Confidence score
    pub confidence: f32,
    /// Signal strength
    pub signal: Signal,
}

/// Trading signal strength
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::StrongBuy => write!(f, "STRONG_BUY"),
            Signal::Buy => write!(f, "BUY"),
            Signal::Hold => write!(f, "HOLD"),
            Signal::Sell => write!(f, "SELL"),
            Signal::StrongSell => write!(f, "STRONG_SELL"),
        }
    }
}

/// Make a prediction from a single sequence.
///
/// # Arguments
///
/// * `model` - Trained GQATrader model
/// * `sequence` - Input sequence of shape (seq_len, 5)
///
/// # Returns
///
/// PredictionResult with prediction, probabilities, and signal
///
/// # Example
///
/// ```rust,no_run
/// use gqa_trading::{GQATrader, predict_next};
/// use ndarray::Array2;
///
/// let model = GQATrader::new(5, 64, 8, 2, 4);
/// let seq = Array2::from_shape_fn((60, 5), |_| rand::random::<f32>());
/// let result = predict_next(&model, &seq);
/// println!("Signal: {}", result.signal);
/// ```
pub fn predict_next(model: &GQATrader, sequence: &Array2<f32>) -> PredictionResult {
    let (prediction, probs) = model.predict_with_probs(sequence);

    let probabilities = [probs[0], probs[1], probs[2]];
    let confidence = calculate_confidence(&probabilities);
    let signal = get_signal(&probabilities, 0.5);

    PredictionResult {
        prediction,
        probabilities,
        confidence,
        signal,
    }
}

/// Make batch predictions.
///
/// # Arguments
///
/// * `model` - Trained GQATrader model
/// * `sequences` - Vector of input sequences
///
/// # Returns
///
/// Vector of predictions
pub fn predict_batch(model: &GQATrader, sequences: &[Array2<f32>]) -> Vec<usize> {
    sequences.iter().map(|seq| model.predict(seq)).collect()
}

/// Make batch predictions with probabilities.
pub fn predict_batch_with_probs(
    model: &GQATrader,
    sequences: &[Array2<f32>],
) -> Vec<PredictionResult> {
    sequences.iter().map(|seq| predict_next(model, seq)).collect()
}

/// Calculate prediction confidence from probabilities.
fn calculate_confidence(probs: &[f32; 3]) -> f32 {
    let mut sorted = *probs;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[0] - sorted[1]
}

/// Get trading signal from probabilities.
fn get_signal(probs: &[f32; 3], threshold: f32) -> Signal {
    let prediction = argmax(probs);
    let confidence = calculate_confidence(probs);

    match prediction {
        2 => {
            if confidence > threshold {
                Signal::StrongBuy
            } else {
                Signal::Buy
            }
        }
        0 => {
            if confidence > threshold {
                Signal::StrongSell
            } else {
                Signal::Sell
            }
        }
        _ => Signal::Hold,
    }
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Analyze a prediction with detailed metrics.
#[derive(Debug, Clone)]
pub struct PredictionAnalysis {
    pub prediction: usize,
    pub prediction_label: String,
    pub probabilities: ProbabilityBreakdown,
    pub confidence: f32,
    pub signal: Signal,
    pub entropy: f32,
    pub recommended_action: String,
}

#[derive(Debug, Clone)]
pub struct ProbabilityBreakdown {
    pub down: f32,
    pub neutral: f32,
    pub up: f32,
}

/// Perform detailed prediction analysis.
pub fn analyze_prediction(model: &GQATrader, sequence: &Array2<f32>) -> PredictionAnalysis {
    let result = predict_next(model, sequence);

    let labels = ["DOWN", "NEUTRAL", "UP"];

    // Calculate entropy
    let entropy: f32 = -result
        .probabilities
        .iter()
        .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
        .sum::<f32>();

    let recommended_action = if result.confidence > 0.3 {
        labels[result.prediction].to_string()
    } else {
        "WAIT".to_string()
    };

    PredictionAnalysis {
        prediction: result.prediction,
        prediction_label: labels[result.prediction].to_string(),
        probabilities: ProbabilityBreakdown {
            down: result.probabilities[0],
            neutral: result.probabilities[1],
            up: result.probabilities[2],
        },
        confidence: result.confidence,
        signal: result.signal,
        entropy,
        recommended_action,
    }
}

/// Ensemble prediction from multiple models.
pub fn ensemble_predict(
    models: &[GQATrader],
    sequence: &Array2<f32>,
    method: EnsembleMethod,
) -> PredictionResult {
    let predictions: Vec<_> = models.iter().map(|m| predict_next(m, sequence)).collect();

    let ensemble_probs = match method {
        EnsembleMethod::Average => {
            let mut avg = [0.0f32; 3];
            for pred in &predictions {
                for i in 0..3 {
                    avg[i] += pred.probabilities[i];
                }
            }
            for v in &mut avg {
                *v /= predictions.len() as f32;
            }
            avg
        }
        EnsembleMethod::Vote => {
            let mut votes = [0.0f32; 3];
            for pred in &predictions {
                votes[pred.prediction] += 1.0;
            }
            for v in &mut votes {
                *v /= predictions.len() as f32;
            }
            votes
        }
        EnsembleMethod::WeightedByConfidence => {
            let total_conf: f32 = predictions.iter().map(|p| p.confidence).sum();
            let mut weighted = [0.0f32; 3];
            for pred in &predictions {
                let weight = pred.confidence / total_conf;
                for i in 0..3 {
                    weighted[i] += pred.probabilities[i] * weight;
                }
            }
            weighted
        }
    };

    let prediction = argmax(&ensemble_probs);
    let confidence = calculate_confidence(&ensemble_probs);
    let signal = get_signal(&ensemble_probs, 0.5);

    PredictionResult {
        prediction,
        probabilities: ensemble_probs,
        confidence,
        signal,
    }
}

/// Ensemble prediction method
#[derive(Debug, Clone, Copy)]
pub enum EnsembleMethod {
    Average,
    Vote,
    WeightedByConfidence,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_predict_next() {
        let model = GQATrader::new(5, 32, 4, 2, 2);
        let seq = Array2::from_shape_fn((30, 5), |_| rand::random::<f32>());

        let result = predict_next(&model, &seq);

        assert!(result.prediction < 3);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

        let prob_sum: f32 = result.probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_predict() {
        let model = GQATrader::new(5, 32, 4, 2, 2);
        let sequences: Vec<_> = (0..10)
            .map(|_| Array2::from_shape_fn((30, 5), |_| rand::random::<f32>()))
            .collect();

        let predictions = predict_batch(&model, &sequences);

        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p < 3));
    }

    #[test]
    fn test_signal_display() {
        assert_eq!(format!("{}", Signal::StrongBuy), "STRONG_BUY");
        assert_eq!(format!("{}", Signal::Hold), "HOLD");
    }
}
