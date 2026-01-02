//! Modern Hopfield Network (2020+)
//!
//! Implementation of the Modern Hopfield Network with continuous states
//! and exponential storage capacity, as described in "Hopfield Networks is All You Need".

use super::{HopfieldConfig, HopfieldNetwork, RetrievalResult};
use ndarray::{Array1, Array2};

/// Modern Hopfield Network with continuous states and attention-like updates
///
/// This implementation follows the paper "Hopfield Networks is All You Need" (2020)
/// which shows that modern Hopfield Networks are equivalent to attention mechanisms.
///
/// # Key Features
///
/// - Continuous states (not binary)
/// - Exponential storage capacity
/// - Softmax-based retrieval (attention mechanism)
/// - Fast convergence
///
/// # Example
///
/// ```rust
/// use hopfield_trading::hopfield::modern::ModernHopfield;
/// use hopfield_trading::hopfield::HopfieldNetwork;
///
/// let mut network = ModernHopfield::new(10);
///
/// // Store patterns
/// let patterns = vec![
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
///     vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
/// ];
/// network.store_patterns(&patterns);
///
/// // Retrieve with a query
/// let query = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.95];
/// let retrieved = network.retrieve(&query);
/// ```
#[derive(Debug, Clone)]
pub struct ModernHopfield {
    /// Stored patterns matrix (each row is a pattern)
    patterns: Array2<f64>,
    /// Network dimension
    dimension: usize,
    /// Number of stored patterns
    num_patterns: usize,
    /// Inverse temperature (beta) - controls sharpness of retrieval
    beta: f64,
    /// Configuration
    config: HopfieldConfig,
    /// Original patterns for reference
    stored_patterns: Vec<Vec<f64>>,
}

impl ModernHopfield {
    /// Create a new Modern Hopfield Network with custom beta
    pub fn with_beta(dimension: usize, beta: f64) -> Self {
        Self {
            patterns: Array2::zeros((0, dimension)),
            dimension,
            num_patterns: 0,
            beta,
            config: HopfieldConfig::default(),
            stored_patterns: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(dimension: usize, config: HopfieldConfig) -> Self {
        Self {
            patterns: Array2::zeros((0, dimension)),
            dimension,
            num_patterns: 0,
            beta: 1.0 / config.temperature,
            config,
            stored_patterns: Vec::new(),
        }
    }

    /// Get the inverse temperature (beta)
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Set the inverse temperature (beta)
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    /// Compute softmax with temperature scaling
    fn softmax(&self, logits: &Array1<f64>) -> Array1<f64> {
        let scaled = logits.mapv(|x| x * self.beta);
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Array1<f64> = scaled.mapv(|x| (x - max_val).exp());
        let sum: f64 = exp_vals.sum();
        exp_vals / sum
    }

    /// Single update step using attention mechanism
    fn update(&self, state: &Array1<f64>) -> Array1<f64> {
        if self.num_patterns == 0 {
            return state.clone();
        }

        // Compute attention scores: softmax(beta * X * state)
        let scores = self.patterns.dot(state); // (num_patterns,)
        let attention = self.softmax(&scores); // (num_patterns,)

        // Compute weighted sum of patterns: X^T * attention
        let mut result = Array1::zeros(self.dimension);
        for (i, &weight) in attention.iter().enumerate() {
            for j in 0..self.dimension {
                result[j] += weight * self.patterns[[i, j]];
            }
        }

        result
    }

    /// Retrieve pattern with detailed result
    pub fn retrieve_detailed(&self, input: &[f64]) -> RetrievalResult {
        let mut state = Array1::from_vec(input.to_vec());
        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            let new_state = self.update(&state);

            iterations = iter + 1;

            // Check for convergence
            let diff: f64 = state
                .iter()
                .zip(new_state.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            state = new_state;

            if diff < self.config.convergence_threshold {
                converged = true;
                break;
            }
        }

        let result_pattern: Vec<f64> = state.to_vec();
        let energy = self.energy(&result_pattern);

        // Find the closest stored pattern
        let (matched_index, confidence) = self.find_closest_pattern(&result_pattern);

        RetrievalResult {
            pattern: result_pattern,
            iterations,
            energy,
            converged,
            matched_index,
            confidence,
        }
    }

    /// Get attention weights for a query
    pub fn attention_weights(&self, query: &[f64]) -> Vec<f64> {
        if self.num_patterns == 0 {
            return Vec::new();
        }

        let q = Array1::from_vec(query.to_vec());
        let scores = self.patterns.dot(&q);
        let attention = self.softmax(&scores);

        attention.to_vec()
    }

    /// Find the closest stored pattern and return (index, confidence)
    fn find_closest_pattern(&self, pattern: &[f64]) -> (Option<usize>, f64) {
        if self.stored_patterns.is_empty() {
            return (None, 0.0);
        }

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (idx, stored) in self.stored_patterns.iter().enumerate() {
            let sim = super::cosine_similarity(pattern, stored);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        // Convert similarity to confidence
        let confidence = (best_sim + 1.0) / 2.0;

        (Some(best_idx), confidence)
    }

    /// Calculate theoretical storage capacity
    ///
    /// Modern Hopfield Networks can store exponentially many patterns
    pub fn capacity(&self) -> String {
        format!("2^(α·{})", self.dimension)
    }

    /// Get stored patterns
    pub fn get_patterns(&self) -> &[Vec<f64>] {
        &self.stored_patterns
    }
}

impl HopfieldNetwork for ModernHopfield {
    fn new(dimension: usize) -> Self {
        Self::with_beta(dimension, 1.0)
    }

    fn store_patterns(&mut self, patterns: &[Vec<f64>]) {
        self.num_patterns = patterns.len();
        self.stored_patterns = patterns.to_vec();

        // Normalize patterns and store in matrix
        let mut pattern_matrix = Array2::zeros((self.num_patterns, self.dimension));

        for (i, pattern) in patterns.iter().enumerate() {
            assert_eq!(
                pattern.len(),
                self.dimension,
                "Pattern dimension mismatch"
            );

            let normalized = super::normalize_pattern(pattern);
            for (j, &val) in normalized.iter().enumerate() {
                pattern_matrix[[i, j]] = val;
            }
        }

        self.patterns = pattern_matrix;
    }

    fn retrieve(&self, input: &[f64]) -> Vec<f64> {
        self.retrieve_detailed(input).pattern
    }

    fn energy(&self, state: &[f64]) -> f64 {
        if self.num_patterns == 0 {
            return 0.0;
        }

        let s = Array1::from_vec(super::normalize_pattern(state));
        let scores = self.patterns.dot(&s);

        // E = -log(Σ exp(x^T * ξ)) + 0.5 * ||x||^2
        let log_sum_exp = {
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = scores.iter().map(|&x| (x - max_score).exp()).sum();
            max_score + sum.ln()
        };

        let norm_sq: f64 = s.iter().map(|x| x.powi(2)).sum();

        -log_sum_exp + 0.5 * norm_sq
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn pattern_count(&self) -> usize {
        self.num_patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut network = ModernHopfield::with_beta(5, 10.0);

        let pattern1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pattern2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        network.store_patterns(&[pattern1.clone(), pattern2.clone()]);

        // Query close to pattern1
        let query = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let result = network.retrieve_detailed(&query);

        assert!(result.matched_index == Some(0));
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_attention_weights() {
        let mut network = ModernHopfield::with_beta(3, 10.0);

        let patterns = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        network.store_patterns(&patterns);

        let query = vec![1.0, 0.0, 0.0];
        let weights = network.attention_weights(&query);

        // First pattern should have highest weight
        assert!(weights[0] > weights[1]);
        assert!(weights[0] > weights[2]);
    }

    #[test]
    fn test_convergence() {
        let mut network = ModernHopfield::with_beta(10, 5.0);

        let patterns: Vec<Vec<f64>> = (0..5)
            .map(|i| (0..10).map(|j| ((i + j) % 3) as f64).collect())
            .collect();

        network.store_patterns(&patterns);

        let query: Vec<f64> = (0..10).map(|x| x as f64 * 0.1).collect();
        let result = network.retrieve_detailed(&query);

        assert!(result.converged || result.iterations < 100);
    }
}
