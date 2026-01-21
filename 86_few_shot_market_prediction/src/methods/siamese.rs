//! Siamese network for few-shot learning
//!
//! Siamese networks learn to compute similarity between pairs of examples.
//! For few-shot classification, queries are compared to support examples.

use super::{FewShotConfig, FewShotLearner, PredictionResult};
use crate::network::{DistanceComputer, EmbeddingNetwork};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Siamese network-based few-shot learner
///
/// Instead of using prototypes, Siamese networks compare the query directly
/// to each support example and aggregate the similarities.
pub struct SiameseLearner {
    config: FewShotConfig,
    network: EmbeddingNetwork,
    distance_computer: DistanceComputer,
    /// Stored support embeddings grouped by class
    support_embeddings: HashMap<usize, Vec<Array1<f64>>>,
    fitted: bool,
}

impl SiameseLearner {
    /// Create a new Siamese learner
    pub fn new(config: FewShotConfig) -> Self {
        let emb_config = config.to_embedding_config();
        let network = EmbeddingNetwork::new(emb_config);
        let distance_computer = DistanceComputer::new(config.distance_metric);

        Self {
            config,
            network,
            distance_computer,
            support_embeddings: HashMap::new(),
            fitted: false,
        }
    }

    /// Compute similarity score between two embeddings
    fn similarity(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        // Use negative distance as similarity
        -self.distance_computer.distance(a, b)
    }

    /// Compute similarity between query and all examples of a class
    fn class_similarity(&self, query: &Array1<f64>, class: usize) -> f64 {
        let class_embs = match self.support_embeddings.get(&class) {
            Some(embs) => embs,
            None => return f64::NEG_INFINITY,
        };

        // Compute mean similarity to all examples in the class
        let total_sim: f64 = class_embs
            .iter()
            .map(|emb| self.similarity(query, emb))
            .sum();

        total_sim / class_embs.len() as f64
    }

    /// Get support embeddings
    pub fn support_embeddings(&self) -> &HashMap<usize, Vec<Array1<f64>>> {
        &self.support_embeddings
    }

    /// Get embedding network
    pub fn network(&self) -> &EmbeddingNetwork {
        &self.network
    }
}

impl FewShotLearner for SiameseLearner {
    fn fit(&mut self, support_features: &Array2<f64>, support_labels: &[usize]) {
        // Embed all support features
        let embeddings = self.network.forward_batch(support_features, None);

        // Group embeddings by class
        self.support_embeddings.clear();
        for (i, &label) in support_labels.iter().enumerate() {
            let emb = embeddings.row(i).to_owned();
            self.support_embeddings.entry(label).or_default().push(emb);
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

        // Get sorted class indices
        let mut class_indices: Vec<_> = self.support_embeddings.keys().cloned().collect();
        class_indices.sort();

        for i in 0..n_queries {
            let query_emb = query_embeddings.row(i).to_owned();

            // Compute similarity to each class
            let similarities: Vec<f64> = class_indices
                .iter()
                .map(|&label| self.class_similarity(&query_emb, label))
                .collect();

            // Convert to probabilities using softmax
            let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sims: Vec<f64> = similarities
                .iter()
                .map(|s| ((s - max_sim) / self.config.temperature).exp())
                .collect();
            let sum: f64 = exp_sims.iter().sum();
            let probs: Vec<f64> = exp_sims.iter().map(|e| e / sum).collect();

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
        let mut class_indices: Vec<_> = self.support_embeddings.keys().cloned().collect();
        class_indices.sort();

        // Compute similarity to each class
        let similarities: Vec<f64> = class_indices
            .iter()
            .map(|&label| self.class_similarity(&query_emb, label))
            .collect();

        // Convert to probabilities
        let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Vec<f64> = similarities
            .iter()
            .map(|s| ((s - max_sim) / self.config.temperature).exp())
            .collect();
        let sum: f64 = exp_sims.iter().sum();
        let probs: Vec<f64> = exp_sims.iter().map(|e| e / sum).collect();

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
    fn test_siamese_learner_fit() {
        let config = FewShotConfig::default().with_input_dim(10);
        let mut learner = SiameseLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        assert!(learner.fitted);
        assert_eq!(learner.support_embeddings.len(), 3);
    }

    #[test]
    fn test_siamese_learner_predict() {
        let config = FewShotConfig::default().with_input_dim(10);
        let mut learner = SiameseLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        let query = Array2::from_shape_vec((2, 10), vec![0.1; 20]).unwrap();
        let predictions = learner.predict(&query);

        assert_eq!(predictions.len(), 2);
        for pred in predictions {
            assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        }
    }
}
