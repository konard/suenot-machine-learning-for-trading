//! Dense Associative Memory implementation
//!
//! Based on "Dense Associative Memory for Pattern Recognition" (Krotov & Hopfield, 2016)
//! and "Hopfield Networks is All You Need" (Ramsauer et al., 2020)

use ndarray::{Array1, Array2, Axis};
use ordered_float::OrderedFloat;

/// Dense Associative Memory for pattern storage and retrieval
#[derive(Debug, Clone)]
pub struct DenseAssociativeMemory {
    /// Stored patterns (memory_size x pattern_dim)
    patterns: Array2<f64>,
    /// Associated values/labels for each pattern
    values: Array1<f64>,
    /// Temperature parameter (inverse of beta)
    beta: f64,
    /// Number of stored patterns
    memory_size: usize,
    /// Pattern dimension
    pattern_dim: usize,
    /// Current number of stored patterns
    current_size: usize,
}

impl DenseAssociativeMemory {
    /// Create a new Dense Associative Memory
    ///
    /// # Arguments
    /// * `memory_size` - Maximum number of patterns to store
    /// * `pattern_dim` - Dimension of each pattern
    /// * `beta` - Temperature parameter (higher = sharper attention)
    pub fn new(memory_size: usize, pattern_dim: usize, beta: f64) -> Self {
        Self {
            patterns: Array2::zeros((memory_size, pattern_dim)),
            values: Array1::zeros(memory_size),
            beta,
            memory_size,
            pattern_dim,
            current_size: 0,
        }
    }

    /// Store patterns in memory
    ///
    /// # Arguments
    /// * `patterns` - Matrix of patterns (n_patterns x pattern_dim)
    /// * `values` - Associated values for each pattern
    pub fn store(&mut self, patterns: &Array2<f64>, values: &Array1<f64>) {
        let (n, dim) = patterns.dim();

        if dim != self.pattern_dim {
            panic!(
                "Pattern dimension mismatch: expected {}, got {}",
                self.pattern_dim, dim
            );
        }

        let to_store = n.min(self.memory_size);
        self.current_size = to_store;

        // Normalize patterns before storing
        for i in 0..to_store {
            let pattern = patterns.row(i);
            let norm: f64 = pattern.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

            for j in 0..self.pattern_dim {
                self.patterns[[i, j]] = if norm > 0.0 {
                    pattern[j] / norm
                } else {
                    pattern[j]
                };
            }
            self.values[i] = values[i];
        }
    }

    /// Store a single pattern
    pub fn store_pattern(&mut self, pattern: &Array1<f64>, value: f64) {
        if self.current_size >= self.memory_size {
            // Memory is full, could implement eviction here
            log::warn!("Memory is full, cannot store more patterns");
            return;
        }

        let norm: f64 = pattern.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        for j in 0..self.pattern_dim {
            self.patterns[[self.current_size, j]] = if norm > 0.0 {
                pattern[j] / norm
            } else {
                pattern[j]
            };
        }
        self.values[self.current_size] = value;
        self.current_size += 1;
    }

    /// Compute similarity between query and all stored patterns
    fn compute_similarities(&self, query: &Array1<f64>) -> Array1<f64> {
        // Normalize query
        let query_norm: f64 = query.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized_query: Array1<f64> = if query_norm > 0.0 {
            query.mapv(|x| x / query_norm)
        } else {
            query.clone()
        };

        // Compute dot products with all stored patterns
        let mut similarities = Array1::zeros(self.current_size);
        for i in 0..self.current_size {
            let dot: f64 = (0..self.pattern_dim)
                .map(|j| normalized_query[j] * self.patterns[[i, j]])
                .sum();
            similarities[i] = dot;
        }

        similarities
    }

    /// Compute softmax attention weights
    fn softmax(&self, similarities: &Array1<f64>) -> Array1<f64> {
        if similarities.is_empty() {
            return Array1::zeros(0);
        }

        // Scaled similarities
        let scaled: Array1<f64> = similarities.mapv(|x| x * self.beta);

        // Numerical stability: subtract max
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Array1<f64> = scaled.mapv(|x| (x - max_val).exp());
        let sum: f64 = exp_vals.sum();

        if sum > 0.0 {
            exp_vals.mapv(|x| x / sum)
        } else {
            Array1::ones(similarities.len()) / similarities.len() as f64
        }
    }

    /// Retrieve the most similar patterns and make a prediction
    ///
    /// # Arguments
    /// * `query` - Query pattern
    ///
    /// # Returns
    /// * `prediction` - Weighted average of values based on similarity
    /// * `confidence` - Confidence score based on attention concentration
    pub fn predict(&self, query: &Array1<f64>) -> (f64, f64) {
        if self.current_size == 0 {
            return (0.0, 0.0);
        }

        let similarities = self.compute_similarities(query);
        let attention = self.softmax(&similarities);

        // Weighted prediction
        let prediction: f64 = (0..self.current_size)
            .map(|i| attention[i] * self.values[i])
            .sum();

        // Confidence based on entropy of attention distribution
        let entropy: f64 = -attention
            .iter()
            .map(|&a| if a > 1e-10 { a * a.ln() } else { 0.0 })
            .sum::<f64>();

        let max_entropy = (self.current_size as f64).ln();
        let confidence = if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            1.0
        };

        (prediction, confidence)
    }

    /// Retrieve top-k similar patterns
    ///
    /// # Returns
    /// Vector of (index, similarity, value) tuples
    pub fn retrieve_top_k(&self, query: &Array1<f64>, k: usize) -> Vec<(usize, f64, f64)> {
        if self.current_size == 0 {
            return Vec::new();
        }

        let similarities = self.compute_similarities(query);

        // Get top-k indices
        let mut indexed: Vec<(usize, f64)> = similarities.iter().cloned().enumerate().collect();
        indexed.sort_by_key(|(_, s)| std::cmp::Reverse(OrderedFloat(*s)));

        indexed
            .into_iter()
            .take(k.min(self.current_size))
            .map(|(idx, sim)| (idx, sim, self.values[idx]))
            .collect()
    }

    /// Get stored pattern by index
    pub fn get_pattern(&self, idx: usize) -> Option<Array1<f64>> {
        if idx >= self.current_size {
            return None;
        }

        Some(self.patterns.row(idx).to_owned())
    }

    /// Get the energy of a state
    ///
    /// E(x) = -log Σ exp(β x · ξ)
    pub fn energy(&self, state: &Array1<f64>) -> f64 {
        if self.current_size == 0 {
            return 0.0;
        }

        let similarities = self.compute_similarities(state);
        let scaled: Array1<f64> = similarities.mapv(|x| x * self.beta);

        // Log-sum-exp for numerical stability
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = scaled.mapv(|x| (x - max_val).exp()).sum();

        -(max_val + sum_exp.ln())
    }

    /// Update state towards stored patterns (Hopfield dynamics)
    pub fn update(&self, state: &Array1<f64>, n_steps: usize) -> Array1<f64> {
        let mut current = state.clone();

        for _ in 0..n_steps {
            let similarities = self.compute_similarities(&current);
            let attention = self.softmax(&similarities);

            // Weighted combination of patterns
            let mut new_state = Array1::zeros(self.pattern_dim);
            for i in 0..self.current_size {
                for j in 0..self.pattern_dim {
                    new_state[j] += attention[i] * self.patterns[[i, j]];
                }
            }

            current = new_state;
        }

        current
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            capacity: self.memory_size,
            current_size: self.current_size,
            pattern_dim: self.pattern_dim,
            beta: self.beta,
            utilization: self.current_size as f64 / self.memory_size as f64,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub capacity: usize,
    pub current_size: usize,
    pub pattern_dim: usize,
    pub beta: f64,
    pub utilization: f64,
}

/// Result of a retrieval operation
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Prediction value
    pub prediction: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Top similar patterns (index, similarity, value)
    pub similar_patterns: Vec<(usize, f64, f64)>,
    /// Average similarity of top patterns
    pub avg_similarity: f64,
}

impl RetrievalResult {
    /// Check if this is a high-confidence retrieval
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Get the direction of prediction
    pub fn direction(&self) -> f64 {
        self.prediction.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut memory = DenseAssociativeMemory::new(10, 4, 1.0);

        let patterns = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let values = Array1::from_vec(vec![1.0, -1.0, 0.5]);

        memory.store(&patterns, &values);

        // Query similar to first pattern
        let query = Array1::from_vec(vec![0.9, 0.1, 0.0, 0.0]);
        let (prediction, confidence) = memory.predict(&query);

        // Should be close to first pattern's value
        assert!(prediction > 0.0);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_top_k_retrieval() {
        let mut memory = DenseAssociativeMemory::new(5, 3, 1.0);

        let patterns = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        memory.store(&patterns, &values);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let top_k = memory.retrieve_top_k(&query, 2);

        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0].0, 0); // First pattern should be most similar
    }

    #[test]
    fn test_energy() {
        let mut memory = DenseAssociativeMemory::new(3, 2, 1.0);

        let patterns = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![1.0, -1.0]);
        memory.store(&patterns, &values);

        // State close to first pattern should have lower energy
        let state1 = Array1::from_vec(vec![0.9, 0.1]);
        let state2 = Array1::from_vec(vec![0.5, 0.5]);

        assert!(memory.energy(&state1) < memory.energy(&state2));
    }
}
