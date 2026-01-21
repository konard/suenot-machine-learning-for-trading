//! Full Context Embeddings (FCE)
//!
//! Implements context-dependent embeddings as described in the Matching Networks paper.
//! FCE allows each embedding to be conditioned on all other embeddings in the set,
//! capturing inter-example relationships.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Full Context Embedding module
///
/// The FCE consists of:
/// 1. Bidirectional LSTM for support set encoding (g')
/// 2. LSTM with read-attention for query encoding (f')
///
/// This implementation uses a simplified attention mechanism
/// for computational efficiency while maintaining the key benefits.
#[derive(Debug, Clone)]
pub struct FullContextEmbedding {
    /// Embedding dimension
    embedding_dim: usize,
    /// Number of processing steps
    num_steps: usize,
    /// Support set encoder weights
    support_weights: FCEWeights,
    /// Query encoder weights
    query_weights: FCEWeights,
}

/// Weights for FCE encoder
#[derive(Debug, Clone)]
struct FCEWeights {
    /// Read attention weights
    attention_weights: Array2<f64>,
    /// LSTM input gate weights
    wi: Array2<f64>,
    /// LSTM forget gate weights
    wf: Array2<f64>,
    /// LSTM output gate weights
    wo: Array2<f64>,
    /// LSTM cell gate weights
    wc: Array2<f64>,
}

impl FullContextEmbedding {
    /// Create a new Full Context Embedding module
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of embeddings
    /// * `num_steps` - Number of read-attention steps (typically 1-5)
    ///
    /// # Example
    /// ```
    /// use matching_networks_finance::network::FullContextEmbedding;
    ///
    /// let fce = FullContextEmbedding::new(32, 3);
    /// ```
    pub fn new(embedding_dim: usize, num_steps: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / embedding_dim as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let mut init_weights = || -> FCEWeights {
            let input_size = embedding_dim * 2; // concatenated input
            FCEWeights {
                attention_weights: Array2::from_shape_fn(
                    (embedding_dim, embedding_dim),
                    |_| rng.sample(&normal),
                ),
                wi: Array2::from_shape_fn((input_size, embedding_dim), |_| rng.sample(&normal)),
                wf: Array2::from_shape_fn((input_size, embedding_dim), |_| rng.sample(&normal)),
                wo: Array2::from_shape_fn((input_size, embedding_dim), |_| rng.sample(&normal)),
                wc: Array2::from_shape_fn((input_size, embedding_dim), |_| rng.sample(&normal)),
            }
        };

        Self {
            embedding_dim,
            num_steps,
            support_weights: init_weights(),
            query_weights: init_weights(),
        }
    }

    /// Encode support set with bidirectional context
    ///
    /// # Arguments
    /// * `embeddings` - Basic embeddings [num_support, embedding_dim]
    ///
    /// # Returns
    /// Context-aware embeddings [num_support, embedding_dim]
    pub fn encode_support(&self, embeddings: &Array2<f64>) -> Array2<f64> {
        if embeddings.nrows() == 0 {
            return embeddings.clone();
        }

        // Simplified bidirectional encoding using mean context
        let _mean_context = embeddings.mean_axis(Axis(0)).unwrap();

        let mut result = Array2::zeros(embeddings.raw_dim());
        for (i, row) in embeddings.rows().into_iter().enumerate() {
            // Exclude current embedding from context
            let context = if embeddings.nrows() > 1 {
                let sum: Array1<f64> = embeddings.sum_axis(Axis(0));
                let adjusted = &sum - &row;
                adjusted / (embeddings.nrows() - 1) as f64
            } else {
                row.to_owned()
            };

            // Combine original embedding with context
            let combined = self.attention_combine(&row.to_owned(), &context);
            for (j, val) in combined.iter().enumerate() {
                result[[i, j]] = *val;
            }
        }

        // L2 normalize
        self.l2_normalize(&result)
    }

    /// Encode query with read-attention over support set
    ///
    /// # Arguments
    /// * `query_embeddings` - Basic query embeddings [num_queries, embedding_dim]
    /// * `support_embeddings` - Context-aware support embeddings [num_support, embedding_dim]
    ///
    /// # Returns
    /// Context-aware query embeddings [num_queries, embedding_dim]
    pub fn encode_query(
        &self,
        query_embeddings: &Array2<f64>,
        support_embeddings: &Array2<f64>,
    ) -> Array2<f64> {
        if query_embeddings.nrows() == 0 || support_embeddings.nrows() == 0 {
            return query_embeddings.clone();
        }

        let mut result = query_embeddings.clone();

        // Iterative refinement with attention
        for _ in 0..self.num_steps {
            let mut new_result = Array2::zeros(result.raw_dim());

            for (i, query) in result.rows().into_iter().enumerate() {
                // Compute attention over support set
                let attention = self.compute_read_attention(&query.to_owned(), support_embeddings);

                // Weighted sum of support embeddings
                let read_context = self.weighted_sum(support_embeddings, &attention);

                // LSTM-style update
                let updated = self.lstm_step(&query.to_owned(), &read_context);

                for (j, val) in updated.iter().enumerate() {
                    new_result[[i, j]] = *val;
                }
            }

            result = new_result;
        }

        // L2 normalize
        self.l2_normalize(&result)
    }

    /// Compute read attention weights
    fn compute_read_attention(
        &self,
        query: &Array1<f64>,
        support: &Array2<f64>,
    ) -> Array1<f64> {
        let num_support = support.nrows();
        let mut scores = Array1::zeros(num_support);

        // Transformed query
        let q_transformed = query.dot(&self.query_weights.attention_weights);

        for (i, s_row) in support.rows().into_iter().enumerate() {
            let score: f64 = q_transformed.iter().zip(s_row.iter()).map(|(a, b)| a * b).sum();
            scores[i] = score;
        }

        // Softmax
        self.softmax(&scores)
    }

    /// Weighted sum of embeddings
    fn weighted_sum(&self, embeddings: &Array2<f64>, weights: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(self.embedding_dim);

        for (row, &weight) in embeddings.rows().into_iter().zip(weights.iter()) {
            for (j, val) in row.iter().enumerate() {
                result[j] += weight * val;
            }
        }

        result
    }

    /// Simplified LSTM step for combining query with read context
    fn lstm_step(&self, query: &Array1<f64>, context: &Array1<f64>) -> Array1<f64> {
        // Concatenate query and context
        let mut input = Vec::with_capacity(self.embedding_dim * 2);
        input.extend(query.iter());
        input.extend(context.iter());
        let input = Array1::from_vec(input);

        // Compute gates (simplified)
        let forget_gate = self.sigmoid(&input.dot(&self.query_weights.wf));
        let input_gate = self.sigmoid(&input.dot(&self.query_weights.wi));
        let output_gate = self.sigmoid(&input.dot(&self.query_weights.wo));
        let cell_candidate = self.tanh(&input.dot(&self.query_weights.wc));

        // Update
        let cell = &forget_gate * query + &input_gate * &cell_candidate;
        let hidden = &output_gate * &self.tanh(&cell);

        hidden
    }

    /// Attention-based combination
    fn attention_combine(&self, embedding: &Array1<f64>, context: &Array1<f64>) -> Array1<f64> {
        // Simple weighted combination with learned attention
        let attention_score = embedding.dot(&self.support_weights.attention_weights.dot(context));
        let alpha = 1.0 / (1.0 + (-attention_score).exp()); // sigmoid

        let combined: Array1<f64> = embedding
            .iter()
            .zip(context.iter())
            .map(|(e, c)| alpha * e + (1.0 - alpha) * c)
            .collect();

        combined
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Tanh activation
    fn tanh(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }

    /// Softmax
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Array1<f64> = x.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_vals.sum();
        exp_vals / sum
    }

    /// L2 normalize embeddings
    fn l2_normalize(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.clone();
        for mut row in result.rows_mut() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-10 {
                row.mapv_inplace(|v| v / norm);
            }
        }
        result
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the number of processing steps
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fce_creation() {
        let fce = FullContextEmbedding::new(32, 3);
        assert_eq!(fce.embedding_dim(), 32);
        assert_eq!(fce.num_steps(), 3);
    }

    #[test]
    fn test_support_encoding() {
        let fce = FullContextEmbedding::new(16, 2);
        let embeddings = Array2::from_shape_fn((5, 16), |_| rand::random::<f64>());

        let encoded = fce.encode_support(&embeddings);

        assert_eq!(encoded.shape(), embeddings.shape());

        // Check L2 normalization
        for row in encoded.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_query_encoding() {
        let fce = FullContextEmbedding::new(16, 2);
        let query = Array2::from_shape_fn((3, 16), |_| rand::random::<f64>());
        let support = Array2::from_shape_fn((5, 16), |_| rand::random::<f64>());

        let encoded_support = fce.encode_support(&support);
        let encoded_query = fce.encode_query(&query, &encoded_support);

        assert_eq!(encoded_query.shape(), query.shape());

        // Check L2 normalization
        for row in encoded_query.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_empty_input() {
        let fce = FullContextEmbedding::new(16, 2);
        let empty = Array2::<f64>::zeros((0, 16));

        let encoded = fce.encode_support(&empty);
        assert_eq!(encoded.shape(), &[0, 16]);
    }
}
