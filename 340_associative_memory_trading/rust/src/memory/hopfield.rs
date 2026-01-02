//! Modern Hopfield Network implementation
//!
//! Based on "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
//! Implements continuous Hopfield networks with exponential storage capacity

use ndarray::{Array1, Array2};

/// Modern Continuous Hopfield Network
///
/// Key properties:
/// - Continuous (real-valued) patterns
/// - Exponential storage capacity
/// - Equivalent to attention mechanism
#[derive(Debug, Clone)]
pub struct ModernHopfield {
    /// Stored patterns (memory_size x pattern_dim)
    patterns: Array2<f64>,
    /// Current number of patterns
    n_patterns: usize,
    /// Pattern dimension
    dim: usize,
    /// Maximum capacity
    capacity: usize,
    /// Inverse temperature (sharpness of retrieval)
    beta: f64,
}

impl ModernHopfield {
    /// Create a new Modern Hopfield Network
    pub fn new(capacity: usize, dim: usize, beta: Option<f64>) -> Self {
        // Default beta scales with sqrt(d) for proper attention
        let default_beta = 1.0 / (dim as f64).sqrt();

        Self {
            patterns: Array2::zeros((capacity, dim)),
            n_patterns: 0,
            dim,
            capacity,
            beta: beta.unwrap_or(default_beta),
        }
    }

    /// Store patterns in the network
    pub fn store(&mut self, patterns: &Array2<f64>) {
        let (n, d) = patterns.dim();

        if d != self.dim {
            panic!("Dimension mismatch: expected {}, got {}", self.dim, d);
        }

        self.n_patterns = n.min(self.capacity);

        for i in 0..self.n_patterns {
            // Store normalized patterns
            let row = patterns.row(i);
            let norm: f64 = row.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

            for j in 0..self.dim {
                self.patterns[[i, j]] = if norm > 0.0 {
                    row[j] / norm
                } else {
                    row[j]
                };
            }
        }
    }

    /// Add a single pattern
    pub fn add_pattern(&mut self, pattern: &Array1<f64>) {
        if self.n_patterns >= self.capacity {
            log::warn!("Hopfield network at capacity");
            return;
        }

        let norm: f64 = pattern.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        for j in 0..self.dim {
            self.patterns[[self.n_patterns, j]] = if norm > 0.0 {
                pattern[j] / norm
            } else {
                pattern[j]
            };
        }

        self.n_patterns += 1;
    }

    /// Compute energy of a state
    ///
    /// E(ξ) = -log Σ_μ exp(β ξ^T ξ_μ)
    ///
    /// Lower energy = closer to stored patterns (attractors)
    pub fn energy(&self, state: &Array1<f64>) -> f64 {
        if self.n_patterns == 0 {
            return 0.0;
        }

        // Normalize state
        let norm: f64 = state.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized: Array1<f64> = if norm > 0.0 {
            state.mapv(|x| x / norm)
        } else {
            state.clone()
        };

        // Compute similarities
        let mut similarities = Vec::with_capacity(self.n_patterns);
        for i in 0..self.n_patterns {
            let dot: f64 = (0..self.dim)
                .map(|j| normalized[j] * self.patterns[[i, j]])
                .sum();
            similarities.push(dot * self.beta);
        }

        // Log-sum-exp
        let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = similarities.iter().map(|&s| (s - max_sim).exp()).sum();

        -(max_sim + sum_exp.ln())
    }

    /// Update state towards attractors using the modern update rule
    ///
    /// ξ_new = Σ_μ softmax(β ξ · ξ_μ) ξ_μ
    pub fn update(&self, state: &Array1<f64>) -> Array1<f64> {
        if self.n_patterns == 0 {
            return state.clone();
        }

        // Normalize state
        let norm: f64 = state.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized: Array1<f64> = if norm > 0.0 {
            state.mapv(|x| x / norm)
        } else {
            state.clone()
        };

        // Compute similarities
        let mut similarities = Array1::zeros(self.n_patterns);
        for i in 0..self.n_patterns {
            let dot: f64 = (0..self.dim)
                .map(|j| normalized[j] * self.patterns[[i, j]])
                .sum();
            similarities[i] = dot * self.beta;
        }

        // Softmax
        let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Array1<f64> = similarities.mapv(|s| (s - max_sim).exp());
        let sum_exp: f64 = exp_sims.sum();
        let attention: Array1<f64> = exp_sims.mapv(|e| e / sum_exp);

        // Weighted combination of patterns
        let mut new_state = Array1::zeros(self.dim);
        for i in 0..self.n_patterns {
            for j in 0..self.dim {
                new_state[j] += attention[i] * self.patterns[[i, j]];
            }
        }

        new_state
    }

    /// Retrieve (converge to nearest stored pattern)
    pub fn retrieve(&self, query: &Array1<f64>, max_steps: usize, tolerance: f64) -> Array1<f64> {
        let mut state = query.clone();

        for _ in 0..max_steps {
            let new_state = self.update(&state);

            // Check convergence
            let diff: f64 = state
                .iter()
                .zip(new_state.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            state = new_state;

            if diff < tolerance {
                break;
            }
        }

        state
    }

    /// Get attention weights for a query
    pub fn attention(&self, query: &Array1<f64>) -> Array1<f64> {
        if self.n_patterns == 0 {
            return Array1::zeros(0);
        }

        // Normalize query
        let norm: f64 = query.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized: Array1<f64> = if norm > 0.0 {
            query.mapv(|x| x / norm)
        } else {
            query.clone()
        };

        // Compute similarities
        let mut similarities = Array1::zeros(self.n_patterns);
        for i in 0..self.n_patterns {
            let dot: f64 = (0..self.dim)
                .map(|j| normalized[j] * self.patterns[[i, j]])
                .sum();
            similarities[i] = dot * self.beta;
        }

        // Softmax
        let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Array1<f64> = similarities.mapv(|s| (s - max_sim).exp());
        let sum_exp: f64 = exp_sims.sum();

        exp_sims.mapv(|e| e / sum_exp)
    }

    /// Get the metastable states count (theoretical capacity)
    ///
    /// Modern Hopfield networks can store exponentially many patterns:
    /// Capacity ∝ exp(β * d / 4)
    pub fn theoretical_capacity(&self) -> f64 {
        (self.beta * self.dim as f64 / 4.0).exp()
    }

    /// Get network statistics
    pub fn stats(&self) -> HopfieldStats {
        HopfieldStats {
            n_patterns: self.n_patterns,
            capacity: self.capacity,
            dim: self.dim,
            beta: self.beta,
            theoretical_capacity: self.theoretical_capacity(),
        }
    }
}

/// Hopfield network statistics
#[derive(Debug, Clone)]
pub struct HopfieldStats {
    pub n_patterns: usize,
    pub capacity: usize,
    pub dim: usize,
    pub beta: f64,
    pub theoretical_capacity: f64,
}

/// Classical Hopfield Network (binary patterns)
///
/// For reference and comparison with modern version
pub struct ClassicalHopfield {
    /// Weight matrix
    weights: Array2<f64>,
    /// Number of neurons
    n: usize,
    /// Stored patterns count
    n_patterns: usize,
}

impl ClassicalHopfield {
    /// Create a new classical Hopfield network
    pub fn new(n: usize) -> Self {
        Self {
            weights: Array2::zeros((n, n)),
            n,
            n_patterns: 0,
        }
    }

    /// Store patterns using Hebbian learning
    ///
    /// W = (1/P) Σ_μ ξ^μ (ξ^μ)^T
    pub fn store(&mut self, patterns: &Array2<f64>) {
        let p = patterns.nrows();
        self.n_patterns = p;

        // Reset weights
        self.weights.fill(0.0);

        // Hebbian learning
        for pat_idx in 0..p {
            for i in 0..self.n {
                for j in 0..self.n {
                    if i != j {
                        self.weights[[i, j]] +=
                            patterns[[pat_idx, i]] * patterns[[pat_idx, j]] / p as f64;
                    }
                }
            }
        }
    }

    /// Update state (asynchronous update)
    pub fn update_async(&self, state: &mut Array1<f64>, idx: usize) {
        let h: f64 = (0..self.n).map(|j| self.weights[[idx, j]] * state[j]).sum();
        state[idx] = h.signum();
    }

    /// Retrieve (converge to attractor)
    pub fn retrieve(&self, initial: &Array1<f64>, max_steps: usize) -> Array1<f64> {
        let mut state = initial.clone();

        for _ in 0..max_steps {
            let old_state = state.clone();

            // Random order update
            for i in 0..self.n {
                self.update_async(&mut state, i);
            }

            // Check convergence
            if state == old_state {
                break;
            }
        }

        state
    }

    /// Compute energy
    pub fn energy(&self, state: &Array1<f64>) -> f64 {
        let mut e = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                e -= 0.5 * self.weights[[i, j]] * state[i] * state[j];
            }
        }
        e
    }

    /// Theoretical capacity (0.14 * N for random patterns)
    pub fn capacity(&self) -> f64 {
        0.14 * self.n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modern_hopfield() {
        let mut network = ModernHopfield::new(10, 4, Some(1.0));

        let patterns = Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        network.store(&patterns);

        // Query close to first pattern
        let query = Array1::from_vec(vec![0.9, 0.1, 0.0, 0.0]);
        let retrieved = network.retrieve(&query, 10, 1e-6);

        // Should converge to first pattern
        assert!((retrieved[0] - 1.0).abs() < 0.1 || retrieved[0] > 0.8);
    }

    #[test]
    fn test_energy_decreases() {
        let mut network = ModernHopfield::new(5, 3, Some(1.0));

        let patterns =
            Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        network.store(&patterns);

        // Random state should have higher energy than stored pattern
        let random = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let stored = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        assert!(network.energy(&random) > network.energy(&stored));
    }

    #[test]
    fn test_attention() {
        let mut network = ModernHopfield::new(3, 2, Some(10.0));

        let patterns = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        network.store(&patterns);

        let query = Array1::from_vec(vec![1.0, 0.0]);
        let attention = network.attention(&query);

        // Should attend mostly to first pattern
        assert!(attention[0] > attention[1]);
    }
}
