//! Prototype Network for market state classification.
//!
//! This module provides inference capabilities for prototype-based
//! classification of market states.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Market state classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    Bullish = 0,
    Bearish = 1,
    Consolidation = 2,
    BreakoutUp = 3,
    BreakoutDown = 4,
}

impl MarketState {
    /// Convert from integer
    pub fn from_int(value: usize) -> Option<Self> {
        match value {
            0 => Some(MarketState::Bullish),
            1 => Some(MarketState::Bearish),
            2 => Some(MarketState::Consolidation),
            3 => Some(MarketState::BreakoutUp),
            4 => Some(MarketState::BreakoutDown),
            _ => None,
        }
    }

    /// Get state name
    pub fn name(&self) -> &'static str {
        match self {
            MarketState::Bullish => "Bullish",
            MarketState::Bearish => "Bearish",
            MarketState::Consolidation => "Consolidation",
            MarketState::BreakoutUp => "BreakoutUp",
            MarketState::BreakoutDown => "BreakoutDown",
        }
    }

    /// Get recommended trading action
    pub fn trading_action(&self) -> TradingAction {
        match self {
            MarketState::Bullish => TradingAction::Long,
            MarketState::Bearish => TradingAction::Short,
            MarketState::Consolidation => TradingAction::Hold,
            MarketState::BreakoutUp => TradingAction::Long,
            MarketState::BreakoutDown => TradingAction::Short,
        }
    }
}

/// Trading action recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAction {
    Long,
    Short,
    Hold,
}

/// Errors that can occur in prototype network
#[derive(Error, Debug)]
pub enum PrototypeError {
    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Prototype not initialized")]
    NotInitialized,

    #[error("Invalid prototype index: {0}")]
    InvalidPrototypeIndex(usize),
}

/// Prototype network for market state classification.
///
/// This is a simplified inference-only implementation that uses
/// pre-trained prototype vectors to classify market states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeNetwork {
    /// Number of input features
    n_features: usize,
    /// Latent dimension
    latent_dim: usize,
    /// Number of classes
    n_classes: usize,
    /// Number of prototypes per class
    prototypes_per_class: usize,
    /// Prototype vectors (n_prototypes x latent_dim)
    prototypes: Array2<f64>,
    /// Encoder weights (simplified linear encoder)
    encoder_weights: Array2<f64>,
    /// Encoder bias
    encoder_bias: Array1<f64>,
    /// Small constant for numerical stability
    epsilon: f64,
}

impl PrototypeNetwork {
    /// Create a new prototype network with random initialization.
    ///
    /// In practice, you would load pre-trained weights.
    pub fn new(
        n_features: usize,
        latent_dim: usize,
        n_classes: usize,
        prototypes_per_class: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let n_prototypes = n_classes * prototypes_per_class;

        // Random initialization (would be replaced by loaded weights)
        let mut prototypes = Array2::zeros((n_prototypes, latent_dim));
        for i in 0..n_prototypes {
            for j in 0..latent_dim {
                prototypes[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        let mut encoder_weights = Array2::zeros((n_features, latent_dim));
        for i in 0..n_features {
            for j in 0..latent_dim {
                encoder_weights[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        let encoder_bias = Array1::zeros(latent_dim);

        Self {
            n_features,
            latent_dim,
            n_classes,
            prototypes_per_class,
            prototypes,
            encoder_weights,
            encoder_bias,
            epsilon: 1e-4,
        }
    }

    /// Load prototype network from serialized data.
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Serialize prototype network to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Get the number of prototypes.
    pub fn num_prototypes(&self) -> usize {
        self.n_classes * self.prototypes_per_class
    }

    /// Get the class for a prototype index.
    pub fn prototype_class(&self, proto_idx: usize) -> usize {
        proto_idx / self.prototypes_per_class
    }

    /// Encode input features to latent space.
    ///
    /// Uses a simplified linear encoder.
    fn encode(&self, features: &Array1<f64>) -> Result<Array1<f64>, PrototypeError> {
        if features.len() != self.n_features {
            return Err(PrototypeError::InvalidShape {
                expected: format!("{}", self.n_features),
                actual: format!("{}", features.len()),
            });
        }

        // Linear transformation: z = features @ weights + bias
        let mut z = self.encoder_bias.clone();
        for j in 0..self.latent_dim {
            for i in 0..self.n_features {
                z[j] += features[i] * self.encoder_weights[[i, j]];
            }
            // ReLU activation
            z[j] = z[j].max(0.0);
        }

        Ok(z)
    }

    /// Compute similarity scores between latent representation and prototypes.
    fn compute_similarities(&self, z: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let n_prototypes = self.num_prototypes();
        let mut distances = Array1::zeros(n_prototypes);
        let mut similarities = Array1::zeros(n_prototypes);

        for p in 0..n_prototypes {
            // Squared L2 distance
            let mut dist_sq = 0.0;
            for j in 0..self.latent_dim {
                let diff = z[j] - self.prototypes[[p, j]];
                dist_sq += diff * diff;
            }
            distances[p] = dist_sq;

            // Convert to similarity using log transformation
            similarities[p] = ((dist_sq + 1.0) / (dist_sq + self.epsilon)).ln();
        }

        (similarities, distances)
    }

    /// Classify based on similarity scores.
    fn classify(&self, similarities: &Array1<f64>) -> (usize, f64) {
        // Sum similarities for each class
        let mut class_scores = Array1::zeros(self.n_classes);

        for p in 0..self.num_prototypes() {
            let class_idx = self.prototype_class(p);
            class_scores[class_idx] += similarities[p];
        }

        // Softmax
        let max_score = class_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_scores = Array1::zeros(self.n_classes);
        let mut sum_exp = 0.0;

        for i in 0..self.n_classes {
            exp_scores[i] = (class_scores[i] - max_score).exp();
            sum_exp += exp_scores[i];
        }

        for i in 0..self.n_classes {
            exp_scores[i] /= sum_exp;
        }

        // Find argmax
        let mut best_class = 0;
        let mut best_prob = exp_scores[0];
        for i in 1..self.n_classes {
            if exp_scores[i] > best_prob {
                best_prob = exp_scores[i];
                best_class = i;
            }
        }

        (best_class, best_prob)
    }

    /// Predict market state for a single sample.
    ///
    /// # Arguments
    /// * `features` - Flattened feature vector of shape (n_features * lookback,)
    ///
    /// # Returns
    /// Tuple of (predicted_state, confidence, similarities)
    pub fn predict_single(
        &self,
        features: &Array1<f64>,
    ) -> Result<(MarketState, f64, Array1<f64>), PrototypeError> {
        // For simplicity, we aggregate features by taking the mean
        // In a full implementation, you would use a conv encoder
        let mut aggregated = Array1::zeros(self.n_features);

        if features.len() % self.n_features != 0 {
            return Err(PrototypeError::InvalidShape {
                expected: format!("multiple of {}", self.n_features),
                actual: format!("{}", features.len()),
            });
        }

        let lookback = features.len() / self.n_features;
        for t in 0..lookback {
            for f in 0..self.n_features {
                aggregated[f] += features[t * self.n_features + f];
            }
        }
        for f in 0..self.n_features {
            aggregated[f] /= lookback as f64;
        }

        // Encode
        let z = self.encode(&aggregated)?;

        // Compute similarities
        let (similarities, _) = self.compute_similarities(&z);

        // Classify
        let (class_idx, confidence) = self.classify(&similarities);

        let state = MarketState::from_int(class_idx)
            .ok_or(PrototypeError::InvalidPrototypeIndex(class_idx))?;

        Ok((state, confidence, similarities))
    }

    /// Predict market states for multiple samples.
    ///
    /// # Arguments
    /// * `features_batch` - Vector of feature arrays
    ///
    /// # Returns
    /// Vector of (predicted_state, confidence) tuples
    pub fn predict(
        &self,
        features_batch: &[Array1<f64>],
    ) -> Result<Vec<(MarketState, f64)>, PrototypeError> {
        features_batch
            .iter()
            .map(|f| {
                let (state, conf, _) = self.predict_single(f)?;
                Ok((state, conf))
            })
            .collect()
    }

    /// Get prototype explanation for a prediction.
    pub fn explain(
        &self,
        features: &Array1<f64>,
    ) -> Result<PrototypeExplanation, PrototypeError> {
        let (state, confidence, similarities) = self.predict_single(features)?;

        // Find most similar prototype
        let mut best_proto_idx = 0;
        let mut best_similarity = similarities[0];
        for i in 1..similarities.len() {
            if similarities[i] > best_similarity {
                best_similarity = similarities[i];
                best_proto_idx = i;
            }
        }

        // Get similarities for each class
        let mut class_similarities = Vec::with_capacity(self.n_classes);
        for c in 0..self.n_classes {
            let start = c * self.prototypes_per_class;
            let end = start + self.prototypes_per_class;
            let avg_sim: f64 = similarities.slice(ndarray::s![start..end]).mean().unwrap_or(0.0);
            class_similarities.push((MarketState::from_int(c).unwrap(), avg_sim));
        }

        Ok(PrototypeExplanation {
            predicted_state: state,
            confidence,
            most_similar_prototype: best_proto_idx,
            prototype_similarity: best_similarity,
            class_similarities,
        })
    }

    /// Set prototypes (for loading pre-trained weights).
    pub fn set_prototypes(&mut self, prototypes: Array2<f64>) -> Result<(), PrototypeError> {
        if prototypes.dim() != (self.num_prototypes(), self.latent_dim) {
            return Err(PrototypeError::InvalidShape {
                expected: format!("({}, {})", self.num_prototypes(), self.latent_dim),
                actual: format!("({}, {})", prototypes.dim().0, prototypes.dim().1),
            });
        }
        self.prototypes = prototypes;
        Ok(())
    }

    /// Set encoder weights (for loading pre-trained weights).
    pub fn set_encoder_weights(
        &mut self,
        weights: Array2<f64>,
        bias: Array1<f64>,
    ) -> Result<(), PrototypeError> {
        if weights.dim() != (self.n_features, self.latent_dim) {
            return Err(PrototypeError::InvalidShape {
                expected: format!("({}, {})", self.n_features, self.latent_dim),
                actual: format!("({}, {})", weights.dim().0, weights.dim().1),
            });
        }
        if bias.len() != self.latent_dim {
            return Err(PrototypeError::InvalidShape {
                expected: format!("{}", self.latent_dim),
                actual: format!("{}", bias.len()),
            });
        }
        self.encoder_weights = weights;
        self.encoder_bias = bias;
        Ok(())
    }
}

/// Explanation of a prototype-based prediction.
#[derive(Debug, Clone)]
pub struct PrototypeExplanation {
    /// Predicted market state
    pub predicted_state: MarketState,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// Index of most similar prototype
    pub most_similar_prototype: usize,
    /// Similarity to most similar prototype
    pub prototype_similarity: f64,
    /// Average similarity for each class
    pub class_similarities: Vec<(MarketState, f64)>,
}

impl PrototypeExplanation {
    /// Generate human-readable explanation.
    pub fn explain(&self) -> String {
        format!(
            "Predicted: {} (confidence: {:.2}%)\n\
             Reasoning: Most similar to prototype {} with similarity {:.4}\n\
             Class similarities: {}",
            self.predicted_state.name(),
            self.confidence * 100.0,
            self.most_similar_prototype,
            self.prototype_similarity,
            self.class_similarities
                .iter()
                .map(|(s, sim)| format!("{}: {:.4}", s.name(), sim))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_state_conversion() {
        assert_eq!(MarketState::from_int(0), Some(MarketState::Bullish));
        assert_eq!(MarketState::from_int(4), Some(MarketState::BreakoutDown));
        assert_eq!(MarketState::from_int(5), None);
    }

    #[test]
    fn test_prototype_network_creation() {
        let net = PrototypeNetwork::new(13, 128, 5, 2);
        assert_eq!(net.num_prototypes(), 10);
        assert_eq!(net.prototype_class(0), 0);
        assert_eq!(net.prototype_class(1), 0);
        assert_eq!(net.prototype_class(2), 1);
    }

    #[test]
    fn test_prediction() {
        let net = PrototypeNetwork::new(13, 128, 5, 2);

        // Create dummy input (13 features * 60 timesteps = 780)
        let features = Array1::from_elem(13 * 60, 0.5);
        let result = net.predict_single(&features);

        assert!(result.is_ok());
        let (state, confidence, _) = result.unwrap();
        assert!(confidence >= 0.0 && confidence <= 1.0);
        assert!(matches!(
            state,
            MarketState::Bullish
                | MarketState::Bearish
                | MarketState::Consolidation
                | MarketState::BreakoutUp
                | MarketState::BreakoutDown
        ));
    }
}
