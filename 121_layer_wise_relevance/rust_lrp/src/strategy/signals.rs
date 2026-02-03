//! Trading signal generation with LRP explanations

use ndarray::Array1;
use crate::model::LRPNetwork;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Signal analysis result
#[derive(Debug, Clone)]
pub struct SignalAnalysis {
    /// Trading signal
    pub signal: Signal,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// Feature relevances
    pub relevances: Vec<(String, f64)>,
    /// Explanation coherence
    pub coherence: f64,
}

/// Signal generator using LRP model
pub struct SignalGenerator {
    /// Feature names
    feature_names: Vec<String>,
    /// Confidence threshold for trading
    confidence_threshold: f64,
}

impl SignalGenerator {
    /// Create new signal generator
    pub fn new(feature_names: Vec<String>, confidence_threshold: f64) -> Self {
        Self {
            feature_names,
            confidence_threshold,
        }
    }

    /// Generate signal from model prediction
    pub fn generate(&self, model: &mut LRPNetwork, input: &Array1<f64>) -> SignalAnalysis {
        // Get prediction
        let probs = model.predict_proba(input);
        let pred_class = if probs[1] > probs[0] { 1 } else { 0 };
        let confidence = probs[pred_class].max(probs[1 - pred_class]);

        // Get explanation
        let relevance = model.explain(input, Some(pred_class));

        // Calculate coherence
        let coherence = calculate_coherence(&relevance);

        // Map relevances to feature names
        let relevances = self.map_relevances(&relevance);

        // Determine signal
        let signal = if confidence >= self.confidence_threshold {
            if pred_class == 1 {
                Signal::Buy
            } else {
                Signal::Sell
            }
        } else {
            Signal::Hold
        };

        SignalAnalysis {
            signal,
            confidence,
            relevances,
            coherence,
        }
    }

    /// Map relevance values to feature names
    fn map_relevances(&self, relevance: &Array1<f64>) -> Vec<(String, f64)> {
        let total: f64 = relevance.iter().map(|r| r.abs()).sum();
        let total = if total < 1e-9 { 1.0 } else { total };

        let n_features = self.feature_names.len();
        let n_timesteps = relevance.len() / n_features;

        // Aggregate over time steps
        let mut feature_relevances = vec![0.0; n_features];
        for t in 0..n_timesteps {
            for f in 0..n_features {
                let idx = t * n_features + f;
                if idx < relevance.len() {
                    feature_relevances[f] += relevance[idx].abs();
                }
            }
        }

        // Normalize and create pairs
        let mut pairs: Vec<(String, f64)> = self.feature_names
            .iter()
            .zip(feature_relevances.iter())
            .map(|(name, &rel)| (name.clone(), rel / total * 100.0))
            .collect();

        // Sort by relevance
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        pairs
    }
}

/// Calculate explanation coherence
///
/// High coherence = few dominant features
/// Low coherence = spread across many features
pub fn calculate_coherence(relevance: &Array1<f64>) -> f64 {
    let abs_rel: Vec<f64> = relevance.iter().map(|r| r.abs()).collect();
    let total: f64 = abs_rel.iter().sum();

    if total < 1e-9 {
        return 0.0;
    }

    // Normalized entropy
    let probs: Vec<f64> = abs_rel.iter().map(|r| r / total).collect();
    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    let max_entropy = (relevance.len() as f64).ln();

    if max_entropy < 1e-9 {
        return 1.0;
    }

    1.0 - (entropy / max_entropy)
}

/// Analyze a single trading signal
pub fn analyze_signal(
    model: &mut LRPNetwork,
    input: &Array1<f64>,
    feature_names: &[String],
) -> SignalAnalysis {
    let generator = SignalGenerator::new(feature_names.to_vec(), 0.55);
    generator.generate(model, input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LRPNetwork, LRPConfig};

    #[test]
    fn test_signal_generation() {
        let config = LRPConfig::default();
        let mut model = LRPNetwork::new(16, vec![8, 4], 2, config);

        let feature_names: Vec<String> = (0..16).map(|i| format!("feature_{}", i)).collect();
        let generator = SignalGenerator::new(feature_names, 0.5);

        let input = Array1::from_vec(vec![0.5; 16]);
        let analysis = generator.generate(&mut model, &input);

        assert!(!analysis.relevances.is_empty());
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
    }
}
