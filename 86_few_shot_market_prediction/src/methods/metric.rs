//! Metric-based few-shot learning (Prototypical Networks style)
//!
//! Classifies queries based on distance to class prototypes computed as
//! centroids of embedded support examples.

use super::{FewShotConfig, FewShotLearner, PredictionResult};
use crate::network::{DistanceComputer, EmbeddingNetwork};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Metric-based few-shot learner
///
/// Uses prototype-based classification where each class is represented
/// by the centroid of its support examples in the embedding space.
pub struct MetricBasedLearner {
    config: FewShotConfig,
    network: EmbeddingNetwork,
    distance_computer: DistanceComputer,
    prototypes: HashMap<usize, Array1<f64>>,
    fitted: bool,
}

impl MetricBasedLearner {
    /// Create a new metric-based learner
    pub fn new(config: FewShotConfig) -> Self {
        let emb_config = config.to_embedding_config();
        let network = EmbeddingNetwork::new(emb_config);
        let distance_computer = DistanceComputer::new(config.distance_metric);

        Self {
            config,
            network,
            distance_computer,
            prototypes: HashMap::new(),
            fitted: false,
        }
    }

    /// Compute prototype for a class from embeddings
    fn compute_prototype(embeddings: &[Array1<f64>]) -> Array1<f64> {
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

    /// Get prototypes
    pub fn prototypes(&self) -> &HashMap<usize, Array1<f64>> {
        &self.prototypes
    }

    /// Get the embedding network
    pub fn network(&self) -> &EmbeddingNetwork {
        &self.network
    }

    /// Embed features using the network
    pub fn embed(&self, features: &Array2<f64>) -> Array2<f64> {
        self.network.forward_batch(features, None)
    }
}

impl FewShotLearner for MetricBasedLearner {
    fn fit(&mut self, support_features: &Array2<f64>, support_labels: &[usize]) {
        // Embed all support features
        let embeddings = self.network.forward_batch(support_features, None);

        // Group embeddings by class
        let mut class_embeddings: HashMap<usize, Vec<Array1<f64>>> = HashMap::new();
        for (i, &label) in support_labels.iter().enumerate() {
            let emb = embeddings.row(i).to_owned();
            class_embeddings.entry(label).or_default().push(emb);
        }

        // Compute prototypes
        self.prototypes.clear();
        for (label, embs) in class_embeddings {
            self.prototypes.insert(label, Self::compute_prototype(&embs));
        }

        self.fitted = true;
    }

    fn predict(&self, query_features: &Array2<f64>) -> Vec<PredictionResult> {
        if !self.fitted {
            panic!("Model not fitted. Call fit() first.");
        }

        let n_queries = query_features.nrows();
        let mut results = Vec::with_capacity(n_queries);

        // Embed query features
        let query_embeddings = self.network.forward_batch(query_features, None);

        // Get sorted class indices for consistent ordering
        let mut class_indices: Vec<_> = self.prototypes.keys().cloned().collect();
        class_indices.sort();

        for i in 0..n_queries {
            let query_emb = query_embeddings.row(i).to_owned();

            // Compute distances to all prototypes
            let distances: Vec<f64> = class_indices
                .iter()
                .map(|&label| {
                    let prototype = &self.prototypes[&label];
                    self.distance_computer.distance(&query_emb, prototype)
                })
                .collect();

            // Convert to probabilities
            let probs = DistanceComputer::distances_to_probs(&distances, self.config.temperature);

            // Find best class
            let (best_idx, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let predicted_class = class_indices[best_idx];

            results.push(PredictionResult::new(
                predicted_class,
                confidence,
                probs,
                self.config.confidence_threshold,
            ));
        }

        results
    }

    fn predict_single(&self, query: &Array1<f64>) -> PredictionResult {
        if !self.fitted {
            panic!("Model not fitted. Call fit() first.");
        }

        // Embed query
        let query_emb = self.network.forward(query, None);

        // Get sorted class indices
        let mut class_indices: Vec<_> = self.prototypes.keys().cloned().collect();
        class_indices.sort();

        // Compute distances to all prototypes
        let distances: Vec<f64> = class_indices
            .iter()
            .map(|&label| {
                let prototype = &self.prototypes[&label];
                self.distance_computer.distance(&query_emb, prototype)
            })
            .collect();

        // Convert to probabilities
        let probs = DistanceComputer::distances_to_probs(&distances, self.config.temperature);

        // Find best class
        let (best_idx, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let predicted_class = class_indices[best_idx];

        PredictionResult::new(
            predicted_class,
            confidence,
            probs,
            self.config.confidence_threshold,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_test_data() -> (Array2<f64>, Vec<usize>) {
        let mut rng = rand::thread_rng();
        let n_classes = 3;
        let n_per_class = 5;
        let dim = 10;

        let mut data = Vec::new();
        let mut labels = Vec::new();

        for class in 0..n_classes {
            let base = class as f64 * 2.0;
            for _ in 0..n_per_class {
                for _ in 0..dim {
                    data.push(base + rng.gen::<f64>() * 0.3);
                }
                labels.push(class);
            }
        }

        let features = Array2::from_shape_vec((n_classes * n_per_class, dim), data).unwrap();
        (features, labels)
    }

    #[test]
    fn test_metric_learner_fit() {
        let config = FewShotConfig::default().with_input_dim(10);
        let mut learner = MetricBasedLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        assert!(learner.fitted);
        assert_eq!(learner.prototypes.len(), 3);
    }

    #[test]
    fn test_metric_learner_predict() {
        let config = FewShotConfig::default().with_input_dim(10);
        let mut learner = MetricBasedLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        let query = Array2::from_shape_vec((2, 10), vec![0.1; 20]).unwrap();
        let predictions = learner.predict(&query);

        assert_eq!(predictions.len(), 2);
        for pred in predictions {
            assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
            assert!((pred.class_probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_metric_learner_predict_single() {
        let config = FewShotConfig::default().with_input_dim(10);
        let mut learner = MetricBasedLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        let query = Array1::from_vec(vec![0.1; 10]);
        let pred = learner.predict_single(&query);

        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }
}
