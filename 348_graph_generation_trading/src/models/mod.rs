//! Graph neural network models for trading.
//!
//! This module provides simplified implementations of graph neural network
//! concepts for trading signal generation.

mod gcn;
mod gat;
mod temporal;

pub use gcn::GraphConvolution;
pub use gat::GraphAttention;
pub use temporal::TemporalGraphModel;

use crate::graph::MarketGraph;
use std::collections::HashMap;

/// Feature vector for a node
pub type NodeFeatures = Vec<f64>;

/// Feature matrix for all nodes
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    /// Features for each symbol
    features: HashMap<String, NodeFeatures>,
    /// Number of features per node
    feature_dim: usize,
}

impl FeatureMatrix {
    /// Create a new feature matrix
    pub fn new(feature_dim: usize) -> Self {
        Self {
            features: HashMap::new(),
            feature_dim,
        }
    }

    /// Create from graph with computed features
    pub fn from_graph(graph: &MarketGraph, returns: &HashMap<String, Vec<f64>>) -> Self {
        let mut matrix = Self::new(5); // 5 features: mean, std, skew, min, max

        for symbol in graph.symbols() {
            if let Some(ret) = returns.get(&symbol) {
                let features = compute_features(ret);
                matrix.set(&symbol, features);
            }
        }

        matrix
    }

    /// Set features for a symbol
    pub fn set(&mut self, symbol: &str, features: NodeFeatures) {
        self.features.insert(symbol.to_string(), features);
    }

    /// Get features for a symbol
    pub fn get(&self, symbol: &str) -> Option<&NodeFeatures> {
        self.features.get(symbol)
    }

    /// Get all symbols with features
    pub fn symbols(&self) -> Vec<String> {
        self.features.keys().cloned().collect()
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Convert to matrix form (ordered by symbols)
    pub fn to_matrix(&self, symbols: &[String]) -> Vec<Vec<f64>> {
        symbols
            .iter()
            .map(|s| {
                self.features
                    .get(s)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.feature_dim])
            })
            .collect()
    }
}

/// Compute statistical features from return series
fn compute_features(returns: &[f64]) -> NodeFeatures {
    if returns.is_empty() {
        return vec![0.0; 5];
    }

    let n = returns.len() as f64;

    // Mean
    let mean: f64 = returns.iter().sum::<f64>() / n;

    // Standard deviation
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    // Skewness
    let skew = if std > 1e-10 {
        let m3: f64 = returns.iter().map(|r| ((r - mean) / std).powi(3)).sum::<f64>() / n;
        m3
    } else {
        0.0
    };

    // Min and max
    let min = returns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    vec![mean, std, skew, min, max]
}

/// Model output: predictions for each symbol
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Predicted values for each symbol
    pub predictions: HashMap<String, f64>,
    /// Confidence scores (optional)
    pub confidence: HashMap<String, f64>,
}

impl ModelOutput {
    /// Create new model output
    pub fn new() -> Self {
        Self {
            predictions: HashMap::new(),
            confidence: HashMap::new(),
        }
    }

    /// Set prediction for a symbol
    pub fn set_prediction(&mut self, symbol: &str, value: f64, confidence: f64) {
        self.predictions.insert(symbol.to_string(), value);
        self.confidence.insert(symbol.to_string(), confidence);
    }

    /// Get top predictions
    pub fn top_predictions(&self, k: usize) -> Vec<(String, f64)> {
        let mut sorted: Vec<_> = self.predictions.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
            .into_iter()
            .take(k)
            .map(|(s, v)| (s.clone(), *v))
            .collect()
    }

    /// Get bottom predictions
    pub fn bottom_predictions(&self, k: usize) -> Vec<(String, f64)> {
        let mut sorted: Vec<_> = self.predictions.iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
            .into_iter()
            .take(k)
            .map(|(s, v)| (s.clone(), *v))
            .collect()
    }
}

impl Default for ModelOutput {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation functions
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f64),
    Linear,
}

impl Activation {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
            Activation::Linear => x,
        }
    }

    /// Apply to vector
    pub fn apply_vec(&self, v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| self.apply(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_features() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];
        let features = compute_features(&returns);

        assert_eq!(features.len(), 5);
        assert!(features[0].abs() < 0.1); // Mean should be small
        assert!(features[1] > 0.0); // Std should be positive
    }

    #[test]
    fn test_feature_matrix() {
        let mut matrix = FeatureMatrix::new(3);
        matrix.set("BTC", vec![0.1, 0.2, 0.3]);
        matrix.set("ETH", vec![0.4, 0.5, 0.6]);

        assert_eq!(matrix.get("BTC"), Some(&vec![0.1, 0.2, 0.3]));
        assert_eq!(matrix.symbols().len(), 2);
    }

    #[test]
    fn test_model_output() {
        let mut output = ModelOutput::new();
        output.set_prediction("BTC", 0.8, 0.9);
        output.set_prediction("ETH", 0.6, 0.85);
        output.set_prediction("SOL", 0.9, 0.7);

        let top = output.top_predictions(2);
        assert_eq!(top[0].0, "SOL");
        assert_eq!(top[1].0, "BTC");
    }

    #[test]
    fn test_activation() {
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(1.0), 1.0);

        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    }
}
