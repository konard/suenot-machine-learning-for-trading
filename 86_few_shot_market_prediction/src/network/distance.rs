//! Distance computation for few-shot learning
//!
//! Provides various distance metrics for comparing embeddings to prototypes.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Distance metric types for prototype comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance: ||a - b||_2
    Euclidean,
    /// Squared Euclidean distance: ||a - b||_2^2
    SquaredEuclidean,
    /// Cosine distance: 1 - cos(a, b)
    Cosine,
    /// Manhattan distance: ||a - b||_1
    Manhattan,
    /// Chebyshev distance: max(|a_i - b_i|)
    Chebyshev,
    /// Mahalanobis distance (requires covariance matrix)
    Mahalanobis,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Euclidean
    }
}

/// Computes distances between embeddings
#[derive(Debug, Clone)]
pub struct DistanceComputer {
    metric: DistanceMetric,
    /// Learned metric matrix for Mahalanobis distance
    metric_matrix: Option<Array2<f64>>,
}

impl DistanceComputer {
    /// Create a new distance computer with specified metric
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            metric,
            metric_matrix: None,
        }
    }

    /// Create with Euclidean distance (default)
    pub fn euclidean() -> Self {
        Self::new(DistanceMetric::Euclidean)
    }

    /// Create with cosine distance
    pub fn cosine() -> Self {
        Self::new(DistanceMetric::Cosine)
    }

    /// Set metric matrix for Mahalanobis distance
    pub fn with_metric_matrix(mut self, matrix: Array2<f64>) -> Self {
        self.metric_matrix = Some(matrix);
        self
    }

    /// Compute distance between two vectors
    pub fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                let diff = a - b;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::SquaredEuclidean => {
                let diff = a - b;
                diff.dot(&diff)
            }
            DistanceMetric::Cosine => {
                let norm_a = a.dot(a).sqrt();
                let norm_b = b.dot(b).sqrt();
                if norm_a < 1e-8 || norm_b < 1e-8 {
                    return 1.0;
                }
                1.0 - a.dot(b) / (norm_a * norm_b)
            }
            DistanceMetric::Manhattan => {
                (a - b).mapv(|x| x.abs()).sum()
            }
            DistanceMetric::Chebyshev => {
                (a - b).mapv(|x| x.abs()).fold(0.0, |acc, &x| acc.max(x))
            }
            DistanceMetric::Mahalanobis => {
                if let Some(ref m) = self.metric_matrix {
                    let diff = a - b;
                    let md = m.dot(&diff);
                    diff.dot(&md).sqrt()
                } else {
                    // Fall back to Euclidean if no metric matrix
                    let diff = a - b;
                    diff.dot(&diff).sqrt()
                }
            }
        }
    }

    /// Compute similarity (inverse of distance, used for softmax)
    pub fn similarity(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.metric {
            DistanceMetric::Cosine => {
                // For cosine, return cosine similarity directly
                let norm_a = a.dot(a).sqrt();
                let norm_b = b.dot(b).sqrt();
                if norm_a < 1e-8 || norm_b < 1e-8 {
                    return 0.0;
                }
                a.dot(b) / (norm_a * norm_b)
            }
            _ => {
                // For distance-based metrics, use negative distance
                -self.distance(a, b)
            }
        }
    }

    /// Compute distances between a query embedding and multiple prototypes
    pub fn distances_to_prototypes(
        &self,
        query: &Array1<f64>,
        prototypes: &[Array1<f64>],
    ) -> Vec<f64> {
        prototypes
            .iter()
            .map(|p| self.distance(query, p))
            .collect()
    }

    /// Compute distance matrix between queries and prototypes
    pub fn distance_matrix(
        &self,
        queries: &Array2<f64>,
        prototypes: &Array2<f64>,
    ) -> Array2<f64> {
        let n_queries = queries.nrows();
        let n_prototypes = prototypes.nrows();
        let mut distances = Array2::zeros((n_queries, n_prototypes));

        for i in 0..n_queries {
            let query = queries.row(i).to_owned();
            for j in 0..n_prototypes {
                let prototype = prototypes.row(j).to_owned();
                distances[[i, j]] = self.distance(&query, &prototype);
            }
        }

        distances
    }

    /// Convert distances to probabilities using softmax
    ///
    /// # Arguments
    /// * `distances` - Distance values (lower = more similar)
    /// * `temperature` - Softmax temperature (higher = more uniform)
    pub fn distances_to_probs(distances: &[f64], temperature: f64) -> Vec<f64> {
        // Use negative distances for softmax (lower distance = higher probability)
        let neg_dists: Vec<f64> = distances.iter().map(|d| -d / temperature).collect();

        // Compute softmax with numerical stability
        let max_val = neg_dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = neg_dists.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();

        exp_vals.iter().map(|e| e / sum).collect()
    }

    /// Get the current metric type
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Compute prototype as centroid of embeddings
pub fn compute_prototype(embeddings: &[Array1<f64>]) -> Array1<f64> {
    if embeddings.is_empty() {
        panic!("Cannot compute prototype from empty embeddings");
    }

    let dim = embeddings[0].len();
    let mut sum = Array1::zeros(dim);

    for emb in embeddings {
        sum = sum + emb;
    }

    sum / embeddings.len() as f64
}

/// Compute prototypes for multiple classes
pub fn compute_class_prototypes(
    embeddings: &Array2<f64>,
    labels: &[usize],
) -> std::collections::HashMap<usize, Array1<f64>> {
    let mut class_embeddings: std::collections::HashMap<usize, Vec<Array1<f64>>> =
        std::collections::HashMap::new();

    for (i, &label) in labels.iter().enumerate() {
        let emb = embeddings.row(i).to_owned();
        class_embeddings.entry(label).or_default().push(emb);
    }

    class_embeddings
        .into_iter()
        .map(|(label, embs)| (label, compute_prototype(&embs)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let computer = DistanceComputer::euclidean();
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let dist = computer.distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let computer = DistanceComputer::cosine();
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);

        // Perpendicular vectors have cosine similarity 0, so distance = 1
        let dist = computer.distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-10);

        // Same direction should have distance 0
        let c = Array1::from_vec(vec![2.0, 0.0]);
        let dist2 = computer.distance(&a, &c);
        assert!(dist2.abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let computer = DistanceComputer::new(DistanceMetric::Manhattan);
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let dist = computer.distance(&a, &b);
        assert!((dist - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_distances_to_probs() {
        let distances = vec![1.0, 2.0, 3.0];
        let probs = DistanceComputer::distances_to_probs(&distances, 1.0);

        // Sum should be 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Closer distance should have higher probability
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_compute_prototype() {
        let embeddings = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0]),
        ];

        let prototype = compute_prototype(&embeddings);
        assert!((prototype[0] - 2.0).abs() < 1e-10);
        assert!((prototype[1] - 3.0).abs() < 1e-10);
        assert!((prototype[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix() {
        let computer = DistanceComputer::euclidean();
        let queries = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let prototypes = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 2.0, 2.0]).unwrap();

        let distances = computer.distance_matrix(&queries, &prototypes);

        assert_eq!(distances.shape(), &[2, 2]);
        assert!(distances[[0, 0]].abs() < 1e-10); // Distance from origin to origin
    }
}
