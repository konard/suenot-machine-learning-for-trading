//! Classical Hopfield Network (1982)
//!
//! Implementation of the original Hopfield Network with binary states
//! and Hebbian learning rule for pattern storage.

use super::{HopfieldConfig, HopfieldNetwork, RetrievalResult};
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;

/// Classical Hopfield Network with binary states
///
/// # Example
///
/// ```rust
/// use hopfield_trading::hopfield::classical::ClassicalHopfield;
/// use hopfield_trading::hopfield::HopfieldNetwork;
///
/// let mut network = ClassicalHopfield::new(10);
///
/// // Store some patterns
/// let patterns = vec![
///     vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
///     vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
/// ];
/// network.store_patterns(&patterns);
///
/// // Retrieve a noisy pattern
/// let noisy = vec![1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
/// let retrieved = network.retrieve(&noisy);
/// ```
#[derive(Debug, Clone)]
pub struct ClassicalHopfield {
    /// Weight matrix
    weights: Array2<f64>,
    /// Network dimension
    dimension: usize,
    /// Stored patterns
    patterns: Vec<Vec<f64>>,
    /// Configuration
    config: HopfieldConfig,
}

impl ClassicalHopfield {
    /// Create a new Classical Hopfield Network with custom configuration
    pub fn with_config(dimension: usize, config: HopfieldConfig) -> Self {
        Self {
            weights: Array2::zeros((dimension, dimension)),
            dimension,
            patterns: Vec::new(),
            config,
        }
    }

    /// Get the weight matrix
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Perform one update step (synchronous)
    fn update_sync(&self, state: &Array1<f64>) -> Array1<f64> {
        let h = self.weights.dot(state);
        h.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 })
    }

    /// Perform one update step (asynchronous - random neuron order)
    fn update_async(&self, state: &mut Array1<f64>, rng: &mut impl Rng) {
        let mut indices: Vec<usize> = (0..self.dimension).collect();
        indices.shuffle(rng);

        for i in indices {
            let h: f64 = (0..self.dimension)
                .map(|j| self.weights[[i, j]] * state[j])
                .sum();
            state[i] = if h >= 0.0 { 1.0 } else { -1.0 };
        }
    }

    /// Retrieve pattern with detailed result
    pub fn retrieve_detailed(&self, input: &[f64]) -> RetrievalResult {
        let mut state = Array1::from_vec(super::binarize_pattern(input));
        let mut rng = rand::thread_rng();

        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            let old_state = state.clone();

            if self.config.async_update {
                self.update_async(&mut state, &mut rng);
            } else {
                state = self.update_sync(&state);
            }

            iterations = iter + 1;

            // Check for convergence
            let diff: f64 = state
                .iter()
                .zip(old_state.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

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

    /// Find the closest stored pattern and return (index, confidence)
    fn find_closest_pattern(&self, pattern: &[f64]) -> (Option<usize>, f64) {
        if self.patterns.is_empty() {
            return (None, 0.0);
        }

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (idx, stored) in self.patterns.iter().enumerate() {
            let sim = super::cosine_similarity(pattern, stored);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        // Convert similarity to confidence (handle negative similarities)
        let confidence = (best_sim + 1.0) / 2.0;

        (Some(best_idx), confidence)
    }

    /// Calculate theoretical storage capacity
    pub fn capacity(&self) -> usize {
        // Classical Hopfield can reliably store ~0.14N patterns
        ((self.dimension as f64) * 0.14) as usize
    }

    /// Check if pattern is stored (is a fixed point)
    pub fn is_stored(&self, pattern: &[f64]) -> bool {
        let state = Array1::from_vec(super::binarize_pattern(pattern));
        let updated = self.update_sync(&state);

        state
            .iter()
            .zip(updated.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    }
}

impl HopfieldNetwork for ClassicalHopfield {
    fn new(dimension: usize) -> Self {
        Self::with_config(dimension, HopfieldConfig::default())
    }

    fn store_patterns(&mut self, patterns: &[Vec<f64>]) {
        // Reset weights
        self.weights = Array2::zeros((self.dimension, self.dimension));
        self.patterns.clear();

        let n = patterns.len() as f64;

        for pattern in patterns {
            assert_eq!(
                pattern.len(),
                self.dimension,
                "Pattern dimension mismatch"
            );

            // Binarize and store
            let binary = super::binarize_pattern(pattern);
            self.patterns.push(binary.clone());

            // Hebbian learning: w_ij += (1/N) * xi * xj
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    if i != j {
                        self.weights[[i, j]] += binary[i] * binary[j] / n;
                    }
                }
            }
        }
    }

    fn retrieve(&self, input: &[f64]) -> Vec<f64> {
        self.retrieve_detailed(input).pattern
    }

    fn energy(&self, state: &[f64]) -> f64 {
        let s = Array1::from_vec(state.to_vec());
        let ws = self.weights.dot(&s);

        // E = -0.5 * s^T * W * s
        -0.5 * s.dot(&ws)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut network = ClassicalHopfield::new(8);

        let pattern1 = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let pattern2 = vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

        network.store_patterns(&[pattern1.clone(), pattern2.clone()]);

        // Retrieve exact pattern
        let retrieved = network.retrieve(&pattern1);
        assert_eq!(retrieved, pattern1);

        // Retrieve with noise (1 bit flip)
        let noisy = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let retrieved_noisy = network.retrieve(&noisy);
        assert_eq!(retrieved_noisy, pattern1);
    }

    #[test]
    fn test_energy_decreases() {
        let mut network = ClassicalHopfield::new(8);
        let pattern = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        network.store_patterns(&[pattern.clone()]);

        // Energy of stored pattern should be lower than random state
        let stored_energy = network.energy(&pattern);
        let random_state = vec![0.5, -0.3, 0.1, -0.8, 0.2, 0.9, -0.4, 0.6];
        let random_energy = network.energy(&super::super::binarize_pattern(&random_state));

        assert!(stored_energy <= random_energy);
    }

    #[test]
    fn test_capacity() {
        let network = ClassicalHopfield::new(100);
        assert_eq!(network.capacity(), 14); // ~0.14 * 100
    }
}
