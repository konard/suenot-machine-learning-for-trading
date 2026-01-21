//! Market regime classifier using prototypical networks
//!
//! Classifies current market conditions into predefined regimes.

use crate::data::MarketRegime;
use crate::network::{DistanceFunction, EmbeddingNetwork, PrototypeComputer};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Result of a classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted market regime
    pub regime: MarketRegime,
    /// Confidence (probability) of the prediction
    pub confidence: f64,
    /// Probabilities for all regimes
    pub probabilities: Vec<(MarketRegime, f64)>,
    /// Distance to the closest prototype
    pub min_distance: f64,
    /// Whether this is considered an outlier (far from all prototypes)
    pub is_outlier: bool,
}

impl ClassificationResult {
    /// Get the second most likely regime
    pub fn second_choice(&self) -> Option<(MarketRegime, f64)> {
        self.probabilities.get(1).cloned()
    }

    /// Check if the classification is uncertain (close probabilities)
    pub fn is_uncertain(&self, threshold: f64) -> bool {
        if let Some((_, second_prob)) = self.second_choice() {
            self.confidence - second_prob < threshold
        } else {
            false
        }
    }
}

/// Market regime classifier using prototypical networks
pub struct RegimeClassifier {
    embedding_network: EmbeddingNetwork,
    prototype_computer: PrototypeComputer,
    distance_fn: DistanceFunction,
    /// Outlier detection threshold
    outlier_threshold: f64,
}

impl RegimeClassifier {
    /// Create a new regime classifier
    pub fn new(
        embedding_network: EmbeddingNetwork,
        distance_fn: DistanceFunction,
    ) -> Self {
        Self {
            embedding_network,
            prototype_computer: PrototypeComputer::new(distance_fn),
            distance_fn,
            outlier_threshold: 10.0,
        }
    }

    /// Set the outlier detection threshold
    pub fn with_outlier_threshold(mut self, threshold: f64) -> Self {
        self.outlier_threshold = threshold;
        self
    }

    /// Initialize prototypes from support set
    ///
    /// # Arguments
    /// * `support_features` - Feature vectors for support set
    /// * `support_labels` - Labels (regime indices) for support set
    pub fn initialize_prototypes(
        &mut self,
        support_features: &Array2<f64>,
        support_labels: &[usize],
    ) {
        // Embed all support features
        let embeddings = self.embedding_network.forward_batch(support_features);

        // Clear existing prototypes
        self.prototype_computer = PrototypeComputer::new(self.distance_fn);

        // Group embeddings by class and add to computer
        let mut class_embeddings: Vec<Vec<Array1<f64>>> = vec![Vec::new(); MarketRegime::count()];

        for (i, &label) in support_labels.iter().enumerate() {
            if label < MarketRegime::count() {
                class_embeddings[label].push(embeddings.row(i).to_owned());
            }
        }

        for (class_idx, embeddings) in class_embeddings.into_iter().enumerate() {
            if !embeddings.is_empty() {
                let n_samples = embeddings.len();
                let dim = embeddings[0].len();
                let mut matrix = Array2::zeros((n_samples, dim));
                for (i, emb) in embeddings.into_iter().enumerate() {
                    matrix.row_mut(i).assign(&emb);
                }
                self.prototype_computer.add_class_examples(class_idx, matrix);
            }
        }

        self.prototype_computer.compute_prototypes();
    }

    /// Classify a single feature vector
    pub fn classify(&self, features: &Array1<f64>) -> ClassificationResult {
        // Embed the features
        let embedding = self.embedding_network.forward(features);

        // Get distances to all prototypes
        let distances = self.prototype_computer.distances_to_prototypes(&embedding);

        // Convert to probabilities using softmax
        let probabilities = self.softmax_from_distances(&distances);

        // Find predicted class
        let (predicted_idx, &min_distance) = distances
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((2, &f64::MAX)); // Default to Sideways

        let regime = MarketRegime::from_index(predicted_idx).unwrap_or(MarketRegime::Sideways);
        let confidence = probabilities.get(predicted_idx).cloned().unwrap_or(0.0);

        // Build sorted probabilities
        let mut probs_with_regimes: Vec<(MarketRegime, f64)> = probabilities
            .iter()
            .enumerate()
            .filter_map(|(idx, &prob)| {
                MarketRegime::from_index(idx).map(|r| (r, prob))
            })
            .collect();
        probs_with_regimes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Check for outlier
        let is_outlier = self.prototype_computer.is_outlier(&embedding, self.outlier_threshold);

        ClassificationResult {
            regime,
            confidence,
            probabilities: probs_with_regimes,
            min_distance,
            is_outlier,
        }
    }

    /// Classify multiple feature vectors at once
    pub fn classify_batch(&self, features: &Array2<f64>) -> Vec<ClassificationResult> {
        (0..features.nrows())
            .map(|i| self.classify(&features.row(i).to_owned()))
            .collect()
    }

    /// Convert distances to probabilities using softmax
    fn softmax_from_distances(&self, distances: &[f64]) -> Vec<f64> {
        // Negate distances (smaller distance = higher probability)
        let neg_distances: Vec<f64> = distances.iter().map(|d| -d).collect();

        // Find max for numerical stability
        let max_val = neg_distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x - max)
        let exp_vals: Vec<f64> = neg_distances.iter().map(|d| (d - max_val).exp()).collect();

        // Normalize
        let sum: f64 = exp_vals.iter().sum();
        if sum > 0.0 {
            exp_vals.iter().map(|v| v / sum).collect()
        } else {
            vec![1.0 / distances.len() as f64; distances.len()]
        }
    }

    /// Get the embedding network
    pub fn embedding_network(&self) -> &EmbeddingNetwork {
        &self.embedding_network
    }

    /// Get mutable reference to embedding network
    pub fn embedding_network_mut(&mut self) -> &mut EmbeddingNetwork {
        &mut self.embedding_network
    }

    /// Get current prototypes (for visualization/debugging)
    pub fn get_prototypes(&self) -> Vec<(MarketRegime, Option<Array1<f64>>)> {
        MarketRegime::all()
            .into_iter()
            .map(|regime| {
                let prototype = self.prototype_computer.get_prototype(regime.to_index());
                (regime, prototype)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::EmbeddingConfig;
    use rand::prelude::*;

    fn create_test_data() -> (Array2<f64>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(42);
        let n_per_class = 10;
        let n_classes = 5;
        let n_features = 10;

        let mut features = Array2::zeros((n_per_class * n_classes, n_features));
        let mut labels = Vec::new();

        for class in 0..n_classes {
            for i in 0..n_per_class {
                let row = class * n_per_class + i;
                for j in 0..n_features {
                    features[[row, j]] = class as f64 + rng.gen::<f64>() * 0.5;
                }
                labels.push(class);
            }
        }

        (features, labels)
    }

    #[test]
    fn test_classifier_initialization() {
        let config = EmbeddingConfig {
            input_dim: 10,
            hidden_dims: vec![16, 8],
            output_dim: 4,
            ..Default::default()
        };
        let network = EmbeddingNetwork::new(config);
        let mut classifier = RegimeClassifier::new(network, DistanceFunction::Euclidean);

        let (features, labels) = create_test_data();
        classifier.initialize_prototypes(&features, &labels);

        // Check that we have prototypes for all classes
        let prototypes = classifier.get_prototypes();
        for (regime, prototype) in prototypes {
            assert!(prototype.is_some(), "Missing prototype for {:?}", regime);
        }
    }

    #[test]
    fn test_classification() {
        let config = EmbeddingConfig {
            input_dim: 10,
            hidden_dims: vec![16, 8],
            output_dim: 4,
            ..Default::default()
        };
        let network = EmbeddingNetwork::new(config);
        let mut classifier = RegimeClassifier::new(network, DistanceFunction::Euclidean);

        let (features, labels) = create_test_data();
        classifier.initialize_prototypes(&features, &labels);

        // Classify a sample from the training data
        let test_features = features.row(0).to_owned();
        let result = classifier.classify(&test_features);

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.probabilities.len(), 5);

        // Probabilities should sum to 1
        let sum: f64 = result.probabilities.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_classification() {
        let config = EmbeddingConfig {
            input_dim: 10,
            hidden_dims: vec![16],
            output_dim: 4,
            ..Default::default()
        };
        let network = EmbeddingNetwork::new(config);
        let mut classifier = RegimeClassifier::new(network, DistanceFunction::Euclidean);

        let (features, labels) = create_test_data();
        classifier.initialize_prototypes(&features, &labels);

        let results = classifier.classify_batch(&features);
        assert_eq!(results.len(), features.nrows());
    }
}
