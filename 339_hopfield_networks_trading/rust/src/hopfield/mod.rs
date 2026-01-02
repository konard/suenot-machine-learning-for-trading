//! Hopfield Network implementations
//!
//! This module provides both Classical (1982) and Modern (2020+) Hopfield Network
//! implementations for pattern storage and retrieval.

pub mod classical;
pub mod modern;

use ndarray::Array1;

/// Common trait for Hopfield Network implementations
pub trait HopfieldNetwork {
    /// Create a new network with specified dimension
    fn new(dimension: usize) -> Self;

    /// Store patterns in the network
    fn store_patterns(&mut self, patterns: &[Vec<f64>]);

    /// Retrieve the closest stored pattern to the input
    fn retrieve(&self, input: &[f64]) -> Vec<f64>;

    /// Get the energy of a given state
    fn energy(&self, state: &[f64]) -> f64;

    /// Get the dimension of the network
    fn dimension(&self) -> usize;

    /// Get the number of stored patterns
    fn pattern_count(&self) -> usize;
}

/// Configuration for Hopfield Network behavior
#[derive(Debug, Clone)]
pub struct HopfieldConfig {
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Temperature parameter for stochastic updates
    pub temperature: f64,
    /// Whether to use asynchronous updates
    pub async_update: bool,
}

impl Default for HopfieldConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            temperature: 1.0,
            async_update: true,
        }
    }
}

/// Result of pattern retrieval
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// The retrieved pattern
    pub pattern: Vec<f64>,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Final energy value
    pub energy: f64,
    /// Whether the network converged
    pub converged: bool,
    /// Index of the closest stored pattern (if identifiable)
    pub matched_index: Option<usize>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Utility functions for pattern manipulation
pub fn normalize_pattern(pattern: &[f64]) -> Vec<f64> {
    let mean: f64 = pattern.iter().sum::<f64>() / pattern.len() as f64;
    let variance: f64 = pattern.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / pattern.len() as f64;
    let std_dev = variance.sqrt().max(1e-10);

    pattern.iter().map(|x| (x - mean) / std_dev).collect()
}

/// Binarize a continuous pattern to {-1, +1}
pub fn binarize_pattern(pattern: &[f64]) -> Vec<f64> {
    pattern.iter().map(|&x| if x >= 0.0 { 1.0 } else { -1.0 }).collect()
}

/// Calculate cosine similarity between two patterns
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Patterns must have same length");

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Calculate Euclidean distance between two patterns
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Patterns must have same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_pattern() {
        let pattern = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_pattern(&pattern);

        // Check mean is approximately 0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Check std dev is approximately 1
        let variance: f64 = normalized.iter().map(|x| x.powi(2)).sum::<f64>() / normalized.len() as f64;
        assert!((variance.sqrt() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_binarize_pattern() {
        let pattern = vec![-0.5, 0.3, -0.1, 0.8, 0.0];
        let binary = binarize_pattern(&pattern);
        assert_eq!(binary, vec![-1.0, 1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = vec![-1.0, 0.0, -1.0];
        assert!((cosine_similarity(&a, &c) + 1.0).abs() < 1e-10);
    }
}
