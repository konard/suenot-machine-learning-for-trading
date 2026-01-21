//! Configuration for few-shot learning methods

use crate::network::{DistanceMetric, EmbeddingConfig};
use serde::{Deserialize, Serialize};

/// Few-shot learning method types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FewShotMethod {
    /// Metric-based learning (Prototypical Networks style)
    Metric,
    /// MAML-inspired gradient-based adaptation
    MAML,
    /// Siamese network for pairwise similarity
    Siamese,
    /// Hybrid combining metric and optimization
    Hybrid,
}

impl Default for FewShotMethod {
    fn default() -> Self {
        Self::Metric
    }
}

/// Configuration for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Few-shot learning method to use
    pub method: FewShotMethod,

    /// Number of classes per episode (N-way)
    pub n_way: usize,

    /// Number of examples per class (K-shot)
    pub k_shot: usize,

    /// Number of query samples per class during training
    pub n_query: usize,

    // Network configuration
    /// Input feature dimension
    pub input_dim: usize,

    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,

    /// Embedding dimension
    pub embedding_dim: usize,

    // Distance/similarity
    /// Distance metric for prototype comparison
    pub distance_metric: DistanceMetric,

    /// Softmax temperature (higher = more uniform distribution)
    pub temperature: f64,

    // MAML-specific
    /// Number of adaptation steps for MAML
    pub adaptation_steps: usize,

    /// Inner loop learning rate for MAML
    pub adaptation_lr: f64,

    // Hybrid-specific
    /// Weight for metric-based predictions in hybrid mode (0-1)
    pub metric_weight: f64,

    // General training
    /// Learning rate for meta-training
    pub learning_rate: f64,

    /// Number of training episodes
    pub n_episodes: usize,

    /// Dropout rate
    pub dropout_rate: f64,

    /// L2 regularization weight
    pub l2_reg: f64,

    // Prediction thresholds
    /// Confidence threshold for reliable predictions
    pub confidence_threshold: f64,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            method: FewShotMethod::Metric,
            n_way: 5,
            k_shot: 5,
            n_query: 10,
            input_dim: 20,
            hidden_dims: vec![128, 64],
            embedding_dim: 32,
            distance_metric: DistanceMetric::Euclidean,
            temperature: 1.0,
            adaptation_steps: 5,
            adaptation_lr: 0.01,
            metric_weight: 0.6,
            learning_rate: 0.001,
            n_episodes: 1000,
            dropout_rate: 0.1,
            l2_reg: 0.0001,
            confidence_threshold: 0.5,
        }
    }
}

impl FewShotConfig {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the few-shot method
    pub fn with_method(mut self, method: FewShotMethod) -> Self {
        self.method = method;
        self
    }

    /// Builder: set N-way
    pub fn with_n_way(mut self, n_way: usize) -> Self {
        self.n_way = n_way;
        self
    }

    /// Builder: set K-shot
    pub fn with_k_shot(mut self, k_shot: usize) -> Self {
        self.k_shot = k_shot;
        self
    }

    /// Builder: set input dimension
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Builder: set embedding dimension
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Builder: set hidden dimensions
    pub fn with_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Builder: set distance metric
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Builder: set temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Builder: set MAML adaptation steps
    pub fn with_adaptation_steps(mut self, steps: usize) -> Self {
        self.adaptation_steps = steps;
        self
    }

    /// Builder: set MAML adaptation learning rate
    pub fn with_adaptation_lr(mut self, lr: f64) -> Self {
        self.adaptation_lr = lr;
        self
    }

    /// Builder: set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Get embedding config derived from this config
    pub fn to_embedding_config(&self) -> EmbeddingConfig {
        EmbeddingConfig {
            input_dim: self.input_dim,
            hidden_dims: self.hidden_dims.clone(),
            output_dim: self.embedding_dim,
            normalize_embeddings: true,
            dropout_rate: self.dropout_rate,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = FewShotConfig::new()
            .with_method(FewShotMethod::MAML)
            .with_n_way(3)
            .with_k_shot(10)
            .with_embedding_dim(64);

        assert_eq!(config.method, FewShotMethod::MAML);
        assert_eq!(config.n_way, 3);
        assert_eq!(config.k_shot, 10);
        assert_eq!(config.embedding_dim, 64);
    }

    #[test]
    fn test_to_embedding_config() {
        let config = FewShotConfig::default();
        let emb_config = config.to_embedding_config();

        assert_eq!(emb_config.input_dim, config.input_dim);
        assert_eq!(emb_config.output_dim, config.embedding_dim);
        assert_eq!(emb_config.hidden_dims, config.hidden_dims);
    }
}
