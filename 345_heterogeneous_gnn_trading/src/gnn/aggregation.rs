//! Semantic aggregation for heterogeneous graphs

use ndarray::Array1;
use super::layers::LinearLayer;

/// Semantic aggregation layer for combining embeddings from different sources
#[derive(Debug)]
pub struct SemanticAggregation {
    /// Hidden dimension
    hidden_dim: usize,
    /// Attention vector for computing importance weights
    attention_vector: Array1<f64>,
    /// Projection for attention computation
    attention_proj: LinearLayer,
}

impl SemanticAggregation {
    /// Create a new semantic aggregation layer
    pub fn new(hidden_dim: usize) -> Self {
        // Initialize attention vector with small random values
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let attention_vector = Array1::from_shape_fn(hidden_dim, |_| {
            (rng.gen::<f64>() - 0.5) * 0.1
        });

        Self {
            hidden_dim,
            attention_vector,
            attention_proj: LinearLayer::new(hidden_dim, hidden_dim),
        }
    }

    /// Aggregate multiple embeddings using learned attention
    pub fn aggregate(&self, embeddings: &[Array1<f64>]) -> Array1<f64> {
        if embeddings.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        if embeddings.len() == 1 {
            return embeddings[0].clone();
        }

        // Compute attention scores for each embedding
        let mut attention_scores = Vec::new();
        for emb in embeddings {
            // Project and compute attention score
            let projected = self.attention_proj.forward(emb);
            let score: f64 = projected.iter()
                .zip(self.attention_vector.iter())
                .map(|(p, a)| p.tanh() * a)
                .sum();
            attention_scores.push(score);
        }

        // Softmax normalization
        let max_score = attention_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = attention_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let attention_weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum
        let mut result = Array1::zeros(self.hidden_dim);
        for (emb, &weight) in embeddings.iter().zip(attention_weights.iter()) {
            result = result + emb * weight;
        }

        result
    }

    /// Aggregate with explicit weights (for interpretability)
    pub fn aggregate_weighted(&self, embeddings: &[Array1<f64>], weights: &[f64]) -> Array1<f64> {
        if embeddings.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        let normalized: Vec<f64> = if sum > 0.0 {
            weights.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / embeddings.len() as f64; embeddings.len()]
        };

        // Weighted sum
        let mut result = Array1::zeros(self.hidden_dim);
        for (emb, &weight) in embeddings.iter().zip(normalized.iter()) {
            result = result + emb * weight;
        }

        result
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(&self, embeddings: &[Array1<f64>]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let mut attention_scores = Vec::new();
        for emb in embeddings {
            let projected = self.attention_proj.forward(emb);
            let score: f64 = projected.iter()
                .zip(self.attention_vector.iter())
                .map(|(p, a)| p.tanh() * a)
                .sum();
            attention_scores.push(score);
        }

        // Softmax
        let max_score = attention_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = attention_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        exp_scores.iter().map(|e| e / sum_exp).collect()
    }
}

/// Hierarchical aggregation for multi-level semantic fusion
#[derive(Debug)]
pub struct HierarchicalAggregation {
    /// First level aggregation (within type)
    level1: SemanticAggregation,
    /// Second level aggregation (across types)
    level2: SemanticAggregation,
}

impl HierarchicalAggregation {
    /// Create a new hierarchical aggregation layer
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            level1: SemanticAggregation::new(hidden_dim),
            level2: SemanticAggregation::new(hidden_dim),
        }
    }

    /// Aggregate embeddings in a hierarchical manner
    /// groups: Vec of (group_name, Vec<embedding>)
    pub fn aggregate(&self, groups: &[(&str, Vec<Array1<f64>>)]) -> Array1<f64> {
        // First level: aggregate within each group
        let group_embeddings: Vec<Array1<f64>> = groups
            .iter()
            .map(|(_, embs)| self.level1.aggregate(embs))
            .collect();

        // Second level: aggregate across groups
        self.level2.aggregate(&group_embeddings)
    }

    /// Get hierarchical attention weights
    pub fn get_attention_weights(&self, groups: &[(&str, Vec<Array1<f64>>)]) -> (Vec<Vec<f64>>, Vec<f64>) {
        // First level weights
        let level1_weights: Vec<Vec<f64>> = groups
            .iter()
            .map(|(_, embs)| self.level1.get_attention_weights(embs))
            .collect();

        // Group embeddings for level 2
        let group_embeddings: Vec<Array1<f64>> = groups
            .iter()
            .map(|(_, embs)| self.level1.aggregate(embs))
            .collect();

        // Second level weights
        let level2_weights = self.level2.get_attention_weights(&group_embeddings);

        (level1_weights, level2_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_aggregation() {
        let agg = SemanticAggregation::new(16);
        let embeddings = vec![
            Array1::ones(16),
            Array1::ones(16) * 2.0,
            Array1::ones(16) * 0.5,
        ];

        let result = agg.aggregate(&embeddings);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_attention_weights() {
        let agg = SemanticAggregation::new(16);
        let embeddings = vec![
            Array1::ones(16),
            Array1::ones(16) * 2.0,
        ];

        let weights = agg.get_attention_weights(&embeddings);
        assert_eq!(weights.len(), 2);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_aggregation() {
        let agg = SemanticAggregation::new(16);
        let embeddings = vec![
            Array1::ones(16),
            Array1::ones(16) * 2.0,
        ];
        let weights = vec![0.3, 0.7];

        let result = agg.aggregate_weighted(&embeddings, &weights);
        assert_eq!(result.len(), 16);
        // Result should be closer to second embedding (higher weight)
        assert!(result[0] > 1.0);
    }

    #[test]
    fn test_hierarchical_aggregation() {
        let agg = HierarchicalAggregation::new(16);
        let groups = vec![
            ("type1", vec![Array1::ones(16), Array1::ones(16) * 2.0]),
            ("type2", vec![Array1::ones(16) * 0.5]),
        ];

        let result = agg.aggregate(&groups);
        assert_eq!(result.len(), 16);
    }
}
