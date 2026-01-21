//! Hybrid few-shot learning combining metric and optimization approaches
//!
//! This approach combines the simplicity of metric-based methods with
//! the adaptability of MAML-style optimization.

use super::{FewShotConfig, FewShotLearner, MAMLLearner, MetricBasedLearner, PredictionResult};
use ndarray::{Array1, Array2};

/// Hybrid few-shot learner
///
/// Combines predictions from metric-based and MAML learners using
/// weighted averaging of their probability distributions.
pub struct HybridLearner {
    config: FewShotConfig,
    metric_learner: MetricBasedLearner,
    maml_learner: MAMLLearner,
    fitted: bool,
}

impl HybridLearner {
    /// Create a new hybrid learner
    pub fn new(config: FewShotConfig) -> Self {
        let metric_learner = MetricBasedLearner::new(config.clone());
        let maml_learner = MAMLLearner::new(config.clone());

        Self {
            config,
            metric_learner,
            maml_learner,
            fitted: false,
        }
    }

    /// Combine probability distributions from two learners
    fn combine_probs(&self, metric_probs: &[f64], maml_probs: &[f64]) -> Vec<f64> {
        let weight = self.config.metric_weight;
        metric_probs
            .iter()
            .zip(maml_probs.iter())
            .map(|(&m, &o)| weight * m + (1.0 - weight) * o)
            .collect()
    }

    /// Get metric learner reference
    pub fn metric_learner(&self) -> &MetricBasedLearner {
        &self.metric_learner
    }

    /// Get MAML learner reference
    pub fn maml_learner(&self) -> &MAMLLearner {
        &self.maml_learner
    }
}

impl FewShotLearner for HybridLearner {
    fn fit(&mut self, support_features: &Array2<f64>, support_labels: &[usize]) {
        // Fit both learners
        self.metric_learner.fit(support_features, support_labels);
        self.maml_learner.fit(support_features, support_labels);
        self.fitted = true;
    }

    fn predict(&self, query_features: &Array2<f64>) -> Vec<PredictionResult> {
        if !self.fitted {
            panic!("Model not fitted. Call fit() first.");
        }

        // Get predictions from both learners
        let metric_results = self.metric_learner.predict(query_features);
        let maml_results = self.maml_learner.predict(query_features);

        // Combine predictions
        let mut results = Vec::with_capacity(query_features.nrows());

        for (metric_pred, maml_pred) in metric_results.iter().zip(maml_results.iter()) {
            // Combine probability distributions
            let combined_probs = self.combine_probs(
                &metric_pred.class_probabilities,
                &maml_pred.class_probabilities,
            );

            // Find best class from combined distribution
            let (best_idx, &confidence) = combined_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            // Map index back to class - assumes same ordering in both learners
            // In a more robust implementation, you'd verify class ordering matches
            let predicted_class = if best_idx < combined_probs.len() {
                // Use metric learner's class mapping
                let metric_prototypes = self.metric_learner.prototypes();
                let mut class_indices: Vec<_> = metric_prototypes.keys().cloned().collect();
                class_indices.sort();
                class_indices.get(best_idx).cloned().unwrap_or(best_idx)
            } else {
                best_idx
            };

            results.push(PredictionResult::new(
                predicted_class,
                confidence,
                combined_probs,
                self.config.confidence_threshold,
            ));
        }

        results
    }

    fn predict_single(&self, query: &Array1<f64>) -> PredictionResult {
        if !self.fitted {
            panic!("Model not fitted. Call fit() first.");
        }

        // Get predictions from both learners
        let metric_pred = self.metric_learner.predict_single(query);
        let maml_pred = self.maml_learner.predict_single(query);

        // Combine probability distributions
        let combined_probs = self.combine_probs(
            &metric_pred.class_probabilities,
            &maml_pred.class_probabilities,
        );

        // Find best class
        let (best_idx, &confidence) = combined_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Get class index
        let metric_prototypes = self.metric_learner.prototypes();
        let mut class_indices: Vec<_> = metric_prototypes.keys().cloned().collect();
        class_indices.sort();
        let predicted_class = class_indices.get(best_idx).cloned().unwrap_or(best_idx);

        PredictionResult::new(
            predicted_class,
            confidence,
            combined_probs,
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
    fn test_hybrid_learner_fit() {
        let config = FewShotConfig::default()
            .with_input_dim(10)
            .with_adaptation_steps(1);
        let mut learner = HybridLearner::new(config);

        let (features, labels) = create_test_data();
        learner.fit(&features, &labels);

        assert!(learner.fitted);
    }

    #[test]
    fn test_hybrid_learner_predict() {
        let config = FewShotConfig::default()
            .with_input_dim(10)
            .with_adaptation_steps(1);
        let mut learner = HybridLearner::new(config);

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
    fn test_combine_probs() {
        let config = FewShotConfig::default().with_input_dim(10);
        let learner = HybridLearner::new(config);

        let metric_probs = vec![0.8, 0.1, 0.1];
        let maml_probs = vec![0.6, 0.3, 0.1];

        let combined = learner.combine_probs(&metric_probs, &maml_probs);

        // With default metric_weight = 0.6
        // combined[0] = 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        assert!((combined[0] - 0.72).abs() < 1e-6);
    }
}
