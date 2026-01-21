//! Few-shot learning methods
//!
//! This module provides multiple few-shot learning approaches:
//! - Metric-based (Prototypical Networks style)
//! - MAML-inspired optimization
//! - Siamese networks for pairwise similarity
//! - Hybrid approaches combining multiple methods

mod metric;
mod maml;
mod siamese;
mod hybrid;
mod config;

pub use metric::MetricBasedLearner;
pub use maml::MAMLLearner;
pub use siamese::SiameseLearner;
pub use hybrid::HybridLearner;
pub use config::{FewShotConfig, FewShotMethod};

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Result of a few-shot prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted class index
    pub predicted_class: usize,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Probabilities for each class
    pub class_probabilities: Vec<f64>,
    /// Whether the prediction is considered reliable
    pub is_reliable: bool,
}

impl PredictionResult {
    /// Create a new prediction result
    pub fn new(
        predicted_class: usize,
        confidence: f64,
        class_probabilities: Vec<f64>,
        confidence_threshold: f64,
    ) -> Self {
        Self {
            predicted_class,
            confidence,
            class_probabilities,
            is_reliable: confidence >= confidence_threshold,
        }
    }

    /// Get entropy of the prediction distribution
    pub fn entropy(&self) -> f64 {
        -self.class_probabilities
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

/// Result of MAML adaptation
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Number of adaptation steps performed
    pub steps: usize,
    /// Loss before adaptation
    pub initial_loss: f64,
    /// Loss after adaptation
    pub final_loss: f64,
    /// Improvement ratio
    pub improvement: f64,
}

impl AdaptationResult {
    /// Create a new adaptation result
    pub fn new(steps: usize, initial_loss: f64, final_loss: f64) -> Self {
        let improvement = if initial_loss > 1e-10 {
            (initial_loss - final_loss) / initial_loss
        } else {
            0.0
        };
        Self {
            steps,
            initial_loss,
            final_loss,
            improvement,
        }
    }
}

/// Trait for few-shot learners
pub trait FewShotLearner {
    /// Fit the learner on a support set
    fn fit(&mut self, support_features: &ndarray::Array2<f64>, support_labels: &[usize]);

    /// Predict on query features
    fn predict(&self, query_features: &ndarray::Array2<f64>) -> Vec<PredictionResult>;

    /// Predict single sample
    fn predict_single(&self, query: &Array1<f64>) -> PredictionResult;
}

/// High-level few-shot predictor that can use different methods
pub struct FewShotPredictor {
    config: FewShotConfig,
    learner: Box<dyn FewShotLearner + Send + Sync>,
    class_names: std::collections::HashMap<usize, String>,
}

impl FewShotPredictor {
    /// Create a new few-shot predictor
    pub fn new(config: FewShotConfig) -> Self {
        let learner: Box<dyn FewShotLearner + Send + Sync> = match config.method {
            FewShotMethod::Metric => Box::new(MetricBasedLearner::new(config.clone())),
            FewShotMethod::MAML => Box::new(MAMLLearner::new(config.clone())),
            FewShotMethod::Siamese => Box::new(SiameseLearner::new(config.clone())),
            FewShotMethod::Hybrid => Box::new(HybridLearner::new(config.clone())),
        };

        Self {
            config,
            learner,
            class_names: std::collections::HashMap::new(),
        }
    }

    /// Set class names for interpretable predictions
    pub fn with_class_names(mut self, names: std::collections::HashMap<usize, String>) -> Self {
        self.class_names = names;
        self
    }

    /// Fit on support set
    pub fn fit(&mut self, support_features: &ndarray::Array2<f64>, support_labels: &[usize]) {
        self.learner.fit(support_features, support_labels);

        // Auto-populate class names if not set
        if self.class_names.is_empty() {
            for &label in support_labels {
                self.class_names
                    .entry(label)
                    .or_insert_with(|| format!("Class_{}", label));
            }
        }
    }

    /// Predict on query features
    pub fn predict(&self, query_features: &ndarray::Array2<f64>) -> Vec<PredictionResult> {
        self.learner.predict(query_features)
    }

    /// Predict single sample
    pub fn predict_single(&self, query: &Array1<f64>) -> PredictionResult {
        self.learner.predict_single(query)
    }

    /// Get class name for a label
    pub fn class_name(&self, label: usize) -> &str {
        self.class_names
            .get(&label)
            .map(|s| s.as_str())
            .unwrap_or("Unknown")
    }

    /// Get configuration
    pub fn config(&self) -> &FewShotConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_synthetic_data() -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
        // Create simple synthetic data for testing
        let n_classes = 3;
        let n_support = 5;
        let n_query = 3;
        let dim = 10;

        let mut support_data = Vec::new();
        let mut support_labels = Vec::new();
        let mut query_data = Vec::new();
        let mut query_labels = Vec::new();

        for class in 0..n_classes {
            let base = class as f64 * 0.5;
            for _ in 0..n_support {
                let sample: Vec<f64> = (0..dim).map(|_| base + rand::random::<f64>() * 0.1).collect();
                support_data.extend(sample);
                support_labels.push(class);
            }
            for _ in 0..n_query {
                let sample: Vec<f64> = (0..dim).map(|_| base + rand::random::<f64>() * 0.1).collect();
                query_data.extend(sample);
                query_labels.push(class);
            }
        }

        let support = Array2::from_shape_vec((n_classes * n_support, dim), support_data).unwrap();
        let query = Array2::from_shape_vec((n_classes * n_query, dim), query_data).unwrap();

        (support, support_labels, query, query_labels)
    }

    #[test]
    fn test_prediction_result() {
        let result = PredictionResult::new(1, 0.8, vec![0.1, 0.8, 0.1], 0.5);
        assert_eq!(result.predicted_class, 1);
        assert!(result.is_reliable);
        assert!(result.entropy() > 0.0);
    }

    #[test]
    fn test_few_shot_predictor_metric() {
        let config = FewShotConfig::default().with_method(FewShotMethod::Metric);
        let mut predictor = FewShotPredictor::new(config);

        let (support, support_labels, query, _) = create_synthetic_data();
        predictor.fit(&support, &support_labels);

        let predictions = predictor.predict(&query);
        assert_eq!(predictions.len(), query.nrows());

        for pred in predictions {
            assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
            assert!((pred.class_probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }
}
