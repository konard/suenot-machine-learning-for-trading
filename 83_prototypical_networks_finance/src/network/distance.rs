//! Distance functions for prototype-based classification

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Distance function types for prototypical networks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// Euclidean distance: sqrt(sum((x-y)^2))
    Euclidean,
    /// Squared Euclidean distance: sum((x-y)^2) - default for prototypical networks
    SquaredEuclidean,
    /// Cosine distance: 1 - (x·y)/(||x|| ||y||)
    Cosine,
    /// Manhattan distance: sum(|x-y|)
    Manhattan,
}

impl Default for DistanceFunction {
    fn default() -> Self {
        Self::SquaredEuclidean
    }
}

impl DistanceFunction {
    /// Compute distance between two vectors
    pub fn compute(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self {
            DistanceFunction::Euclidean => Self::euclidean(a, b),
            DistanceFunction::SquaredEuclidean => Self::squared_euclidean(a, b),
            DistanceFunction::Cosine => Self::cosine(a, b),
            DistanceFunction::Manhattan => Self::manhattan(a, b),
        }
    }

    /// Euclidean distance
    fn euclidean(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        Self::squared_euclidean(a, b).sqrt()
    }

    /// Squared Euclidean distance (default for prototypical networks)
    fn squared_euclidean(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let diff = a - b;
        diff.dot(&diff)
    }

    /// Cosine distance
    fn cosine(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a < 1e-8 || norm_b < 1e-8 {
            return 1.0; // Maximum distance if either vector is zero
        }

        1.0 - (dot / (norm_a * norm_b))
    }

    /// Manhattan distance
    fn manhattan(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        (a - b).mapv(|x| x.abs()).sum()
    }

    /// Compute softmax probabilities from negative distances
    pub fn distances_to_probabilities(distances: &Array1<f64>, temperature: f64) -> Array1<f64> {
        // Negative distances because smaller distance = higher probability
        let neg_distances = -distances / temperature;

        // Numerical stability: subtract max
        let max_val = neg_distances.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals = neg_distances.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_vals.sum();

        if sum_exp < 1e-8 {
            // Uniform distribution if all distances are very large
            Array1::from_elem(distances.len(), 1.0 / distances.len() as f64)
        } else {
            exp_vals / sum_exp
        }
    }
}

/// Alias for backward compatibility
pub type DistanceType = DistanceFunction;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_squared_euclidean() {
        let df = DistanceFunction::SquaredEuclidean;
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let dist = df.compute(&a, &b);
        assert_relative_eq!(dist, 27.0, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean() {
        let df = DistanceFunction::Euclidean;
        let a = array![0.0, 0.0, 0.0];
        let b = array![3.0, 4.0, 0.0];

        let dist = df.compute(&a, &b);
        assert_relative_eq!(dist, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine() {
        let df = DistanceFunction::Cosine;
        let a = array![1.0, 0.0];
        let b = array![1.0, 0.0];

        // Same direction = 0 distance
        let dist = df.compute(&a, &b);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6);

        // Orthogonal = 1 distance
        let c = array![0.0, 1.0];
        let dist2 = df.compute(&a, &c);
        assert_relative_eq!(dist2, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax() {
        let distances = array![1.0, 2.0, 3.0];

        let probs = DistanceFunction::distances_to_probabilities(&distances, 1.0);

        // Should sum to 1
        assert_relative_eq!(probs.sum(), 1.0, epsilon = 1e-6);

        // Smallest distance should have highest probability
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }
}
