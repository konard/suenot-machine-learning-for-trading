//! Prototype computation and management for prototypical networks

use ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::distance::DistanceFunction;

/// A prototype representing a class centroid in embedding space
#[derive(Debug, Clone)]
pub struct Prototype {
    /// Class identifier
    pub class_id: usize,
    /// Embedding vector (centroid)
    pub embedding: Array1<f64>,
    /// Number of examples used to compute this prototype
    pub n_examples: usize,
    /// Optional variance for confidence estimation
    pub variance: Option<f64>,
}

impl Prototype {
    /// Create a new prototype
    pub fn new(class_id: usize, embedding: Array1<f64>, n_examples: usize) -> Self {
        Self {
            class_id,
            embedding,
            n_examples,
            variance: None,
        }
    }

    /// Create prototype with variance
    pub fn with_variance(mut self, variance: f64) -> Self {
        self.variance = Some(variance);
        self
    }
}

/// Computes and manages prototypes from support set embeddings
#[derive(Debug, Clone)]
pub struct PrototypeComputer {
    /// Distance function for classification
    distance_fn: DistanceFunction,
    /// Stored class embeddings before computing prototypes
    class_embeddings: HashMap<usize, Vec<Array1<f64>>>,
    /// Computed prototypes
    prototypes: HashMap<usize, Prototype>,
}

impl PrototypeComputer {
    /// Create a new prototype computer
    pub fn new(distance_fn: DistanceFunction) -> Self {
        Self {
            distance_fn,
            class_embeddings: HashMap::new(),
            prototypes: HashMap::new(),
        }
    }

    /// Add embeddings for a specific class
    pub fn add_class_examples(&mut self, class_idx: usize, embeddings: Array2<f64>) {
        let entry = self.class_embeddings.entry(class_idx).or_insert_with(Vec::new);
        for i in 0..embeddings.nrows() {
            entry.push(embeddings.row(i).to_owned());
        }
    }

    /// Compute prototypes from all added class examples
    pub fn compute_prototypes(&mut self) {
        self.prototypes.clear();

        for (&class_id, class_embs) in &self.class_embeddings {
            if class_embs.is_empty() {
                continue;
            }

            let embed_dim = class_embs[0].len();

            // Compute centroid (mean)
            let mut centroid = Array1::zeros(embed_dim);
            for emb in class_embs {
                centroid = centroid + emb;
            }
            centroid = centroid / class_embs.len() as f64;

            // Compute variance for confidence estimation
            let variance = if class_embs.len() > 1 {
                let mut var_sum = 0.0f64;
                for emb in class_embs {
                    let diff = emb - &centroid;
                    var_sum += diff.dot(&diff);
                }
                Some(var_sum / class_embs.len() as f64)
            } else {
                None
            };

            let mut prototype = Prototype::new(class_id, centroid, class_embs.len());
            if let Some(v) = variance {
                prototype = prototype.with_variance(v);
            }
            self.prototypes.insert(class_id, prototype);
        }
    }

    /// Get a prototype for a specific class
    pub fn get_prototype(&self, class_idx: usize) -> Option<Array1<f64>> {
        self.prototypes.get(&class_idx).map(|p| p.embedding.clone())
    }

    /// Get all computed prototypes
    pub fn get_all_prototypes(&self) -> Vec<&Prototype> {
        let mut protos: Vec<_> = self.prototypes.values().collect();
        protos.sort_by_key(|p| p.class_id);
        protos
    }

    /// Compute distances from a query to all prototypes
    pub fn distances_to_prototypes(&self, query: &Array1<f64>) -> Vec<f64> {
        let mut distances = Vec::new();
        let max_class = self.prototypes.keys().max().copied().unwrap_or(0);

        for class_idx in 0..=max_class {
            if let Some(proto) = self.prototypes.get(&class_idx) {
                distances.push(self.distance_fn.compute(query, &proto.embedding));
            } else {
                distances.push(f64::MAX);
            }
        }

        distances
    }

    /// Classify a single query embedding
    /// Returns (predicted_class, probabilities)
    pub fn classify(&self, query: &Array1<f64>) -> (usize, Vec<f64>) {
        let distances = self.distances_to_prototypes(query);

        // Convert distances to probabilities using softmax on negative distances
        let neg_distances: Vec<f64> = distances.iter().map(|d| -d).collect();
        let max_neg = neg_distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let exp_vals: Vec<f64> = neg_distances.iter().map(|d| (d - max_neg).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        let probabilities: Vec<f64> = exp_vals.iter().map(|v| v / sum).collect();

        // Find class with highest probability
        let (pred_idx, _) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));

        (pred_idx, probabilities)
    }

    /// Check if a query is an outlier (far from all prototypes)
    pub fn is_outlier(&self, query: &Array1<f64>, threshold: f64) -> bool {
        let distances = self.distances_to_prototypes(query);
        let min_distance = distances.iter().cloned().fold(f64::INFINITY, f64::min);
        min_distance > threshold
    }

    /// Get minimum distance to any prototype
    pub fn min_distance(&self, query: &Array1<f64>) -> f64 {
        let distances = self.distances_to_prototypes(query);
        distances.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Clear all stored data
    pub fn clear(&mut self) {
        self.class_embeddings.clear();
        self.prototypes.clear();
    }
}

impl Default for PrototypeComputer {
    fn default() -> Self {
        Self::new(DistanceFunction::Euclidean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_prototype_computation() {
        let mut computer = PrototypeComputer::default();

        // Add embeddings for class 0
        let class0_embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0,
                1.0, 0.1, 0.0, 0.0,
                1.0, -0.1, 0.0, 0.0,
            ],
        ).unwrap();
        computer.add_class_examples(0, class0_embeddings);

        // Add embeddings for class 1
        let class1_embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.1, 0.0,
                0.0, 1.0, -0.1, 0.0,
            ],
        ).unwrap();
        computer.add_class_examples(1, class1_embeddings);

        computer.compute_prototypes();

        let protos = computer.get_all_prototypes();
        assert_eq!(protos.len(), 2);
        assert_eq!(protos[0].class_id, 0);
        assert_eq!(protos[1].class_id, 1);
        assert_eq!(protos[0].n_examples, 3);
    }

    #[test]
    fn test_classification() {
        let mut computer = PrototypeComputer::default();

        // Add embeddings
        let class0 = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let class1 = Array2::from_shape_vec((1, 4), vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        computer.add_class_examples(0, class0);
        computer.add_class_examples(1, class1);
        computer.compute_prototypes();

        // Query close to class 0
        let query = array![0.9, 0.1, 0.0, 0.0];
        let (pred, probs) = computer.classify(&query);

        assert_eq!(pred, 0);
        assert!(probs[0] > probs[1]);
    }

    #[test]
    fn test_outlier_detection() {
        let mut computer = PrototypeComputer::default();

        let class0 = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        computer.add_class_examples(0, class0);
        computer.compute_prototypes();

        // Normal point
        let normal = array![0.1, 0.1];
        assert!(!computer.is_outlier(&normal, 1.0));

        // Outlier
        let outlier = array![10.0, 10.0];
        assert!(computer.is_outlier(&outlier, 1.0));
    }
}
