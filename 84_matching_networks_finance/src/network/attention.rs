//! Attention Module for Matching Networks
//!
//! Computes attention weights between query and support set embeddings
//! using various distance/similarity functions.

use ndarray::{Array1, Array2, Axis};

/// Distance/Similarity functions for attention computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceFunction {
    /// Cosine similarity (default in original paper)
    Cosine,
    /// Euclidean distance (converted to similarity)
    Euclidean,
    /// Dot product similarity
    DotProduct,
    /// Learned distance (requires additional parameters)
    Learned,
}

impl Default for DistanceFunction {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Attention module for computing weighted voting
#[derive(Debug, Clone)]
pub struct AttentionModule {
    /// Distance function to use
    distance_fn: DistanceFunction,
    /// Temperature for softmax (lower = sharper attention)
    temperature: f64,
    /// Learned distance weights (if using Learned distance)
    learned_weights: Option<Array1<f64>>,
}

impl AttentionModule {
    /// Create a new attention module
    ///
    /// # Arguments
    /// * `distance_fn` - Distance function to use
    /// * `temperature` - Softmax temperature (default: 1.0)
    ///
    /// # Example
    /// ```
    /// use matching_networks_finance::network::{AttentionModule, DistanceFunction};
    ///
    /// let attention = AttentionModule::new(DistanceFunction::Cosine, 1.0);
    /// ```
    pub fn new(distance_fn: DistanceFunction, temperature: f64) -> Self {
        Self {
            distance_fn,
            temperature,
            learned_weights: None,
        }
    }

    /// Create an attention module with default settings
    pub fn default_module() -> Self {
        Self::new(DistanceFunction::Cosine, 1.0)
    }

    /// Set learned weights for Learned distance function
    pub fn with_learned_weights(mut self, weights: Array1<f64>) -> Self {
        self.learned_weights = Some(weights);
        self
    }

    /// Compute attention weights between query and support embeddings
    ///
    /// # Arguments
    /// * `query_embeddings` - Query embeddings [num_queries, embedding_dim]
    /// * `support_embeddings` - Support embeddings [num_support, embedding_dim]
    ///
    /// # Returns
    /// Attention weights [num_queries, num_support] where each row sums to 1
    pub fn compute_attention(
        &self,
        query_embeddings: &Array2<f64>,
        support_embeddings: &Array2<f64>,
    ) -> Array2<f64> {
        // Compute similarities/distances
        let similarities = match self.distance_fn {
            DistanceFunction::Cosine => self.cosine_similarity(query_embeddings, support_embeddings),
            DistanceFunction::Euclidean => {
                self.euclidean_to_similarity(query_embeddings, support_embeddings)
            }
            DistanceFunction::DotProduct => {
                self.dot_product_similarity(query_embeddings, support_embeddings)
            }
            DistanceFunction::Learned => {
                self.learned_similarity(query_embeddings, support_embeddings)
            }
        };

        // Apply temperature scaling and softmax
        self.softmax_with_temperature(&similarities)
    }

    /// Compute attention for a single query
    pub fn compute_attention_single(
        &self,
        query_embedding: &Array1<f64>,
        support_embeddings: &Array2<f64>,
    ) -> Array1<f64> {
        let query_2d = query_embedding.clone().insert_axis(Axis(0));
        let attention = self.compute_attention(&query_2d, support_embeddings);
        attention.index_axis(Axis(0), 0).to_owned()
    }

    /// Cosine similarity between query and support embeddings
    fn cosine_similarity(
        &self,
        query: &Array2<f64>,
        support: &Array2<f64>,
    ) -> Array2<f64> {
        let num_queries = query.nrows();
        let num_support = support.nrows();

        let mut similarities = Array2::zeros((num_queries, num_support));

        for (i, q_row) in query.rows().into_iter().enumerate() {
            for (j, s_row) in support.rows().into_iter().enumerate() {
                let dot: f64 = q_row.iter().zip(s_row.iter()).map(|(a, b)| a * b).sum();
                let q_norm: f64 = q_row.iter().map(|v| v * v).sum::<f64>().sqrt();
                let s_norm: f64 = s_row.iter().map(|v| v * v).sum::<f64>().sqrt();

                let similarity = if q_norm > 1e-10 && s_norm > 1e-10 {
                    dot / (q_norm * s_norm)
                } else {
                    0.0
                };

                similarities[[i, j]] = similarity;
            }
        }

        similarities
    }

    /// Euclidean distance converted to similarity
    fn euclidean_to_similarity(
        &self,
        query: &Array2<f64>,
        support: &Array2<f64>,
    ) -> Array2<f64> {
        let num_queries = query.nrows();
        let num_support = support.nrows();

        let mut similarities = Array2::zeros((num_queries, num_support));

        for (i, q_row) in query.rows().into_iter().enumerate() {
            for (j, s_row) in support.rows().into_iter().enumerate() {
                let distance: f64 = q_row
                    .iter()
                    .zip(s_row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                // Convert distance to similarity (negative distance)
                similarities[[i, j]] = -distance;
            }
        }

        similarities
    }

    /// Dot product similarity
    fn dot_product_similarity(
        &self,
        query: &Array2<f64>,
        support: &Array2<f64>,
    ) -> Array2<f64> {
        query.dot(&support.t())
    }

    /// Learned similarity using weighted distance
    fn learned_similarity(
        &self,
        query: &Array2<f64>,
        support: &Array2<f64>,
    ) -> Array2<f64> {
        match &self.learned_weights {
            Some(weights) => {
                let num_queries = query.nrows();
                let num_support = support.nrows();
                let mut similarities = Array2::zeros((num_queries, num_support));

                for (i, q_row) in query.rows().into_iter().enumerate() {
                    for (j, s_row) in support.rows().into_iter().enumerate() {
                        let weighted_diff: f64 = q_row
                            .iter()
                            .zip(s_row.iter())
                            .zip(weights.iter())
                            .map(|((a, b), w)| w * (a - b).powi(2))
                            .sum();
                        similarities[[i, j]] = -weighted_diff.sqrt();
                    }
                }

                similarities
            }
            None => self.cosine_similarity(query, support),
        }
    }

    /// Softmax with temperature scaling
    fn softmax_with_temperature(&self, logits: &Array2<f64>) -> Array2<f64> {
        let scaled = logits / self.temperature;

        let mut result = Array2::zeros(scaled.raw_dim());

        for (i, row) in scaled.rows().into_iter().enumerate() {
            // Numerical stability: subtract max
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = row.iter().map(|v| (v - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();

            for (j, exp_val) in exp_vals.into_iter().enumerate() {
                result[[i, j]] = exp_val / sum;
            }
        }

        result
    }

    /// Get the temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set the temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Get the distance function
    pub fn distance_function(&self) -> DistanceFunction {
        self.distance_fn
    }
}

impl Default for AttentionModule {
    fn default() -> Self {
        Self::default_module()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cosine_attention() {
        let attention = AttentionModule::new(DistanceFunction::Cosine, 1.0);

        // Create identical vectors - should have similarity 1.0
        let query = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap();
        let support = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let weights = attention.compute_attention(&query, &support);

        assert_eq!(weights.shape(), &[1, 2]);
        // First support should get higher weight (same direction as query)
        assert!(weights[[0, 0]] > weights[[0, 1]]);
        // Sum should be 1
        assert_relative_eq!(weights.sum(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let attention = AttentionModule::new(DistanceFunction::Euclidean, 1.0);

        let query = Array2::from_shape_fn((3, 5), |_| rand::random::<f64>());
        let support = Array2::from_shape_fn((7, 5), |_| rand::random::<f64>());

        let weights = attention.compute_attention(&query, &support);

        assert_eq!(weights.shape(), &[3, 7]);

        // Each row should sum to 1
        for row in weights.rows() {
            assert_relative_eq!(row.sum(), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_temperature_effect() {
        let low_temp = AttentionModule::new(DistanceFunction::Cosine, 0.1);
        let high_temp = AttentionModule::new(DistanceFunction::Cosine, 10.0);

        let query = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let support = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0]).unwrap();

        let low_weights = low_temp.compute_attention(&query, &support);
        let high_weights = high_temp.compute_attention(&query, &support);

        // Low temperature should produce sharper (more concentrated) attention
        let low_max = low_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let high_max = high_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(low_max > high_max, "Low temperature should have sharper attention");
    }

    #[test]
    fn test_single_attention() {
        let attention = AttentionModule::default();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let support = Array2::from_shape_vec((3, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        let weights = attention.compute_attention_single(&query, &support);

        assert_eq!(weights.len(), 3);
        assert_relative_eq!(weights.sum(), 1.0, epsilon = 1e-6);
    }
}
