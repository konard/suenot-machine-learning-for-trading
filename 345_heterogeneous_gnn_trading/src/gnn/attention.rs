//! Attention mechanisms for Heterogeneous GNN

use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};

use super::layers::LinearLayer;
use crate::graph::EdgeFeatures;

/// Type-specific projection layer
#[derive(Debug, Clone)]
pub struct TypeProjection {
    /// Projection weight matrix
    weights: Array2<f64>,
    /// Bias vector
    bias: Array1<f64>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl TypeProjection {
    /// Create a new type projection
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Project a single feature vector
    pub fn project(&self, input: &Array1<f64>) -> Array1<f64> {
        // Handle dimension mismatch by padding/truncating
        let mut adjusted_input = Array1::zeros(self.input_dim);
        for (i, &val) in input.iter().enumerate() {
            if i < self.input_dim {
                adjusted_input[i] = val;
            }
        }
        self.weights.dot(&adjusted_input) + &self.bias
    }

    /// Project a batch of feature vectors
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let n = input.nrows();
        let mut output = Array2::zeros((n, self.output_dim));

        for i in 0..n {
            let row = input.row(i).to_owned();
            let projected = self.project(&row);
            for (j, &val) in projected.iter().enumerate() {
                output[[i, j]] = val;
            }
        }

        output
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }
}

/// Relation-specific attention mechanism
#[derive(Debug)]
pub struct RelationAttention {
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Query projection
    query_proj: LinearLayer,
    /// Key projection
    key_proj: LinearLayer,
    /// Value projection
    value_proj: LinearLayer,
    /// Output projection
    output_proj: LinearLayer,
    /// Edge feature projection
    edge_proj: LinearLayer,
}

impl RelationAttention {
    /// Create a new relation attention layer
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;

        Self {
            hidden_dim,
            num_heads,
            head_dim,
            query_proj: LinearLayer::new(hidden_dim, hidden_dim),
            key_proj: LinearLayer::new(hidden_dim, hidden_dim),
            value_proj: LinearLayer::new(hidden_dim, hidden_dim),
            output_proj: LinearLayer::new(hidden_dim, hidden_dim),
            edge_proj: LinearLayer::new(8, hidden_dim),  // Edge features to hidden dim
        }
    }

    /// Compute attention between source and target nodes
    pub fn compute_attention(
        &self,
        query: &Array1<f64>,
        key: &Array1<f64>,
        edge_features: &EdgeFeatures,
    ) -> f64 {
        let q = self.query_proj.forward(query);
        let k = self.key_proj.forward(key);

        // Dot product attention with edge features
        let edge_vec = self.edge_to_vector(edge_features);
        let edge_bias = self.edge_proj.forward(&edge_vec);

        let attention_score: f64 = q.iter()
            .zip(k.iter())
            .zip(edge_bias.iter())
            .map(|((q_i, k_i), e_i)| q_i * k_i + e_i * 0.1)
            .sum();

        // Scale by sqrt(d_k)
        attention_score / (self.hidden_dim as f64).sqrt()
    }

    /// Compute message from neighbor
    pub fn compute_message(
        &self,
        query: &Array1<f64>,
        neighbor: &Array1<f64>,
        edge_features: &EdgeFeatures,
    ) -> Array1<f64> {
        let attention = self.compute_attention(query, neighbor, edge_features);
        let attention_weight = 1.0 / (1.0 + (-attention).exp());  // Sigmoid normalization

        let value = self.value_proj.forward(neighbor);
        value.mapv(|x| x * attention_weight)
    }

    /// Convert edge features to vector
    fn edge_to_vector(&self, features: &EdgeFeatures) -> Array1<f64> {
        Array1::from(vec![
            features.weight,
            features.confidence,
            features.correlation.unwrap_or(0.0),
            features.lead_lag_seconds.unwrap_or(0) as f64 / 3600.0,
            features.volume.unwrap_or(0.0).ln().max(0.0) / 20.0,
            features.amount.unwrap_or(0.0).ln().max(0.0) / 20.0,
            (features.timestamp as f64) / 1e15,
            0.0,  // Padding
        ])
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.query_proj.param_count()
            + self.key_proj.param_count()
            + self.value_proj.param_count()
            + self.output_proj.param_count()
            + self.edge_proj.param_count()
    }
}

/// Multi-head attention mechanism
#[derive(Debug)]
pub struct MultiHeadAttention {
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Query projection per head
    query_weights: Vec<Array2<f64>>,
    /// Key projection per head
    key_weights: Vec<Array2<f64>>,
    /// Value projection per head
    value_weights: Vec<Array2<f64>>,
    /// Output projection
    output_proj: LinearLayer,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let mut rng = rand::thread_rng();
        let std = (2.0 / (hidden_dim + head_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut query_weights = Vec::new();
        let mut key_weights = Vec::new();
        let mut value_weights = Vec::new();

        for _ in 0..num_heads {
            query_weights.push(Array2::from_shape_fn((head_dim, hidden_dim), |_| normal.sample(&mut rng)));
            key_weights.push(Array2::from_shape_fn((head_dim, hidden_dim), |_| normal.sample(&mut rng)));
            value_weights.push(Array2::from_shape_fn((head_dim, hidden_dim), |_| normal.sample(&mut rng)));
        }

        Self {
            hidden_dim,
            num_heads,
            head_dim,
            query_weights,
            key_weights,
            value_weights,
            output_proj: LinearLayer::new(hidden_dim, hidden_dim),
        }
    }

    /// Compute multi-head attention
    pub fn forward(
        &self,
        query: &Array1<f64>,
        keys: &[Array1<f64>],
        values: &[Array1<f64>],
    ) -> Array1<f64> {
        if keys.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        let mut head_outputs = Vec::new();

        for head in 0..self.num_heads {
            // Project query
            let q = self.query_weights[head].dot(query);

            // Compute attention scores
            let mut scores = Vec::new();
            for key in keys {
                let k = self.key_weights[head].dot(key);
                let score: f64 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                scores.push(score / (self.head_dim as f64).sqrt());
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let attention_weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            let mut head_output = Array1::zeros(self.head_dim);
            for (i, value) in values.iter().enumerate() {
                let v = self.value_weights[head].dot(value);
                head_output = head_output + v * attention_weights[i];
            }

            head_outputs.push(head_output);
        }

        // Concatenate heads
        let mut concatenated = Array1::zeros(self.hidden_dim);
        for (head, output) in head_outputs.iter().enumerate() {
            for (i, &val) in output.iter().enumerate() {
                concatenated[head * self.head_dim + i] = val;
            }
        }

        // Output projection
        self.output_proj.forward(&concatenated)
    }

    /// Aggregate multiple embeddings using attention
    pub fn aggregate(&self, embeddings: &[Array1<f64>]) -> Array1<f64> {
        if embeddings.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        if embeddings.len() == 1 {
            return embeddings[0].clone();
        }

        // Use mean as query
        let n = embeddings.len() as f64;
        let query = embeddings.iter().fold(
            Array1::zeros(self.hidden_dim),
            |acc, e| acc + e,
        ) / n;

        self.forward(&query, embeddings, embeddings)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.num_heads * self.head_dim * self.hidden_dim * 3 + self.output_proj.param_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_projection() {
        let proj = TypeProjection::new(10, 16);
        let input = Array1::ones(10);
        let output = proj.project(&input);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_relation_attention() {
        let attn = RelationAttention::new(16, 4);
        let query = Array1::ones(16);
        let key = Array1::ones(16) * 0.5;
        let edge_features = EdgeFeatures::with_correlation(0.8, 1000);

        let score = attn.compute_attention(&query, &key, &edge_features);
        assert!(score.is_finite());
    }

    #[test]
    fn test_multi_head_attention() {
        let attn = MultiHeadAttention::new(16, 4);
        let query = Array1::ones(16);
        let keys = vec![Array1::ones(16), Array1::ones(16) * 0.5];
        let values = vec![Array1::ones(16), Array1::ones(16) * 2.0];

        let output = attn.forward(&query, &keys, &values);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_aggregate() {
        let attn = MultiHeadAttention::new(16, 4);
        let embeddings = vec![
            Array1::ones(16),
            Array1::ones(16) * 2.0,
            Array1::ones(16) * 0.5,
        ];

        let output = attn.aggregate(&embeddings);
        assert_eq!(output.len(), 16);
    }
}
