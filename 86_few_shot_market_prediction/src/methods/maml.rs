//! MAML-inspired few-shot learning
//!
//! Model-Agnostic Meta-Learning uses gradient-based adaptation for quick
//! learning on new tasks. This implementation uses a simplified version
//! with finite-difference gradient approximation.

use super::{AdaptationResult, FewShotConfig, FewShotLearner, PredictionResult};
use crate::network::{DistanceComputer, EmbeddingNetwork, NetworkParams};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// MAML-inspired few-shot learner
///
/// Performs gradient-based adaptation on the support set before making
/// predictions. This allows the model to quickly adapt to new task distributions.
pub struct MAMLLearner {
    config: FewShotConfig,
    network: EmbeddingNetwork,
    distance_computer: DistanceComputer,
    /// Adapted parameters after fitting
    adapted_params: Option<NetworkParams>,
    /// Prototypes computed with adapted parameters
    prototypes: HashMap<usize, Array1<f64>>,
    /// Last adaptation result
    last_adaptation: Option<AdaptationResult>,
    fitted: bool,
}

impl MAMLLearner {
    /// Create a new MAML learner
    pub fn new(config: FewShotConfig) -> Self {
        let emb_config = config.to_embedding_config();
        let network = EmbeddingNetwork::new(emb_config);
        let distance_computer = DistanceComputer::new(config.distance_metric);

        Self {
            config,
            network,
            distance_computer,
            adapted_params: None,
            prototypes: HashMap::new(),
            last_adaptation: None,
            fitted: false,
        }
    }

    /// Compute prototype-based loss for a set of embeddings and labels
    fn compute_loss(
        &self,
        embeddings: &Array2<f64>,
        labels: &[usize],
        temperature: f64,
    ) -> f64 {
        // Compute prototypes
        let mut class_embeddings: HashMap<usize, Vec<Array1<f64>>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            let emb = embeddings.row(i).to_owned();
            class_embeddings.entry(label).or_default().push(emb);
        }

        let prototypes: HashMap<usize, Array1<f64>> = class_embeddings
            .into_iter()
            .map(|(label, embs)| {
                let dim = embs[0].len();
                let mut sum = Array1::zeros(dim);
                for emb in &embs {
                    sum = sum + emb;
                }
                (label, sum / embs.len() as f64)
            })
            .collect();

        // Compute loss as negative log probability of correct class
        let mut class_indices: Vec<_> = prototypes.keys().cloned().collect();
        class_indices.sort();

        let mut total_loss = 0.0;
        for (i, &true_label) in labels.iter().enumerate() {
            let emb = embeddings.row(i).to_owned();

            // Compute distances to all prototypes
            let distances: Vec<f64> = class_indices
                .iter()
                .map(|&label| {
                    let prototype = &prototypes[&label];
                    self.distance_computer.distance(&emb, prototype)
                })
                .collect();

            // Convert to probabilities
            let probs = DistanceComputer::distances_to_probs(&distances, temperature);

            // Find index of true class
            let true_idx = class_indices.iter().position(|&l| l == true_label).unwrap();

            // Negative log probability
            total_loss -= (probs[true_idx] + 1e-10).ln();
        }

        total_loss / labels.len() as f64
    }

    /// Perform gradient-based adaptation using finite differences
    ///
    /// This is a simplified implementation. In practice, you would use
    /// automatic differentiation for better performance.
    fn adapt(
        &self,
        support_features: &Array2<f64>,
        support_labels: &[usize],
    ) -> (NetworkParams, AdaptationResult) {
        let mut params = self.network.clone_params();
        let eps = 1e-5;

        // Compute initial loss
        let initial_embeddings = self.network.forward_batch(support_features, Some(&params));
        let initial_loss = self.compute_loss(&initial_embeddings, support_labels, self.config.temperature);

        // Perform adaptation steps
        for _step in 0..self.config.adaptation_steps {
            // Compute gradients using finite differences (simplified)
            let mut weight_grads: Vec<Array2<f64>> = Vec::new();
            let mut bias_grads: Vec<Array1<f64>> = Vec::new();

            // For each layer, compute approximate gradients
            for layer_idx in 0..params.weights.len() {
                let (rows, cols) = params.weights[layer_idx].dim();
                let mut w_grad = Array2::zeros((rows, cols));

                // Sample random indices for stochastic gradient approximation
                // (full gradient would be too expensive)
                let n_samples = std::cmp::min(10, rows * cols);
                let mut rng = rand::thread_rng();

                for _ in 0..n_samples {
                    let i = rand::Rng::gen_range(&mut rng, 0..rows);
                    let j = rand::Rng::gen_range(&mut rng, 0..cols);

                    // Finite difference approximation
                    params.weights[layer_idx][[i, j]] += eps;
                    let emb_plus = self.network.forward_batch(support_features, Some(&params));
                    let loss_plus = self.compute_loss(&emb_plus, support_labels, self.config.temperature);
                    params.weights[layer_idx][[i, j]] -= eps;

                    w_grad[[i, j]] = (loss_plus - initial_loss) / eps;
                }
                weight_grads.push(w_grad);

                // Bias gradients
                let bias_len = params.biases[layer_idx].len();
                let mut b_grad = Array1::zeros(bias_len);

                for idx in 0..std::cmp::min(5, bias_len) {
                    params.biases[layer_idx][idx] += eps;
                    let emb_plus = self.network.forward_batch(support_features, Some(&params));
                    let loss_plus = self.compute_loss(&emb_plus, support_labels, self.config.temperature);
                    params.biases[layer_idx][idx] -= eps;

                    b_grad[idx] = (loss_plus - initial_loss) / eps;
                }
                bias_grads.push(b_grad);
            }

            // Apply gradients
            params.apply_gradients(&weight_grads, &bias_grads, self.config.adaptation_lr);
        }

        // Compute final loss
        let final_embeddings = self.network.forward_batch(support_features, Some(&params));
        let final_loss = self.compute_loss(&final_embeddings, support_labels, self.config.temperature);

        let result = AdaptationResult::new(self.config.adaptation_steps, initial_loss, final_loss);

        (params, result)
    }

    /// Get last adaptation result
    pub fn last_adaptation(&self) -> Option<&AdaptationResult> {
        self.last_adaptation.as_ref()
    }

    /// Get prototypes
    pub fn prototypes(&self) -> &HashMap<usize, Array1<f64>> {
        &self.prototypes
    }
}

impl FewShotLearner for MAMLLearner {
    fn fit(&mut self, support_features: &Array2<f64>, support_labels: &[usize]) {
        // Perform adaptation
        let (adapted_params, adaptation_result) = self.adapt(support_features, support_labels);

        // Store adapted parameters
        self.adapted_params = Some(adapted_params.clone());
        self.last_adaptation = Some(adaptation_result);

        // Compute prototypes using adapted parameters
        let embeddings = self.network.forward_batch(support_features, Some(&adapted_params));

        let mut class_embeddings: HashMap<usize, Vec<Array1<f64>>> = HashMap::new();
        for (i, &label) in support_labels.iter().enumerate() {
            let emb = embeddings.row(i).to_owned();
            class_embeddings.entry(label).or_default().push(emb);
        }

        self.prototypes.clear();
        for (label, embs) in class_embeddings {
            let dim = embs[0].len();
            let mut sum = Array1::zeros(dim);
            for emb in &embs {
                sum = sum + emb;
            }
            self.prototypes.insert(label, sum / embs.len() as f64);
        }

        self.fitted = true;
    }

    fn predict(&self, query_features: &Array2<f64>) -> Vec<PredictionResult> {
        if !self.fitted {
            panic!("Model not fitted. Call fit() first.");
        }

        let params = self.adapted_params.as_ref().unwrap();
        let n_queries = query_features.nrows();
        let mut results = Vec::with_capacity(n_queries);

        // Embed query features using adapted parameters
        let query_embeddings = self.network.forward_batch(query_features, Some(params));

        // Get sorted class indices
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

        let params = self.adapted_params.as_ref().unwrap();

        // Embed query using adapted parameters
        let query_emb = self.network.forward(query, Some(params));

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
    fn test_maml_learner_fit() {
        let config = FewShotConfig::default()
            .with_input_dim(10)
            .with_adaptation_steps(2);
        let mut learner = MAMLLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        assert!(learner.fitted);
        assert!(learner.adapted_params.is_some());
        assert!(learner.last_adaptation.is_some());
    }

    #[test]
    fn test_maml_learner_predict() {
        let config = FewShotConfig::default()
            .with_input_dim(10)
            .with_adaptation_steps(1);
        let mut learner = MAMLLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        let query = Array2::from_shape_vec((2, 10), vec![0.1; 20]).unwrap();
        let predictions = learner.predict(&query);

        assert_eq!(predictions.len(), 2);
    }
}
